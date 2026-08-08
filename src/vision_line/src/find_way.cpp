#include <opencv2/opencv.hpp>
#include "camera_capture.h"
#include "ground_perspective.h"
#include "image_process.h"
#include <chrono>
#include <thread>

enum ProcessState
{
    IDLE = 0,
    STRAIGHT_TRACKING = 1,
    RIGHT_TURNING = 2,
    LEFT_TURNING = 3,

};

const char *state_name(ProcessState s)
{
    switch (s)
    {
    case IDLE:
        return "IDLE";
    case STRAIGHT_TRACKING:
        return "STRAIGHT_TRACKING";
    case RIGHT_TURNING:
        return "RIGHT_TURNING";
    case LEFT_TURNING:
        return "LEFT_TURNING";
    default:
        return "UNKNOWN";
    }
}

struct InitSignal
{
    bool straight_signal = false;
    bool left_signal = false;
    bool right_signal = false;
};

int track_target_p(const std::vector<cv::Point> &line_points, int target_index, int w, int h)
{
    if (line_points.empty() || w <= 0 || h <= 0)
    {
        // std::cout << "Invalid input for track_target_p" << std::endl;
        return -1000;
    }

    if (static_cast<int>(line_points.size()) < std::abs(target_index) + 1)
    {
        // std::cout << "Index out of bounds for track_target_p" << std::endl;
        return -1001;
    }

    // 处理负索引
    int idx = target_index;
    if (idx < 0)
    {
        idx = static_cast<int>(line_points.size()) + idx;
    }
    if (idx < 0 || idx >= static_cast<int>(line_points.size()))
    {
        // std::cout << "Index out of bounds for track_target_p" << std::endl;
        return -1000;
    }
    cv::Point pt = line_points[idx];
    double x_raw = pt.x;
    double y_raw = pt.y;
    double x_error = x_raw - (w / 2.0);
    std::cout << "Tracking target point: target idx:" << target_index << ", y:" << y_raw << ",x_error = " << x_error << std::endl;
    return x_error;
}

MidLineMode midLineModeForSide(SearchSide side)
{
    if (side == LEFT_ONLY)
        return LEFT_OFFSET;
    if (side == RIGHT_ONLY)
        return RIGHT_OFFSET;
    return MID_AVG;
}

SearchSide oppositeSide(SearchSide side)
{
    if (side == LEFT_ONLY)
        return RIGHT_ONLY;
    if (side == RIGHT_ONLY)
        return LEFT_ONLY;
    return BOTH;
}

bool cornerDetected(ImageProcess &img_process, SearchSide side)
{
    const bool left_found = img_process.get_left_corners() != cv::Point(0, 0);
    const bool right_found = img_process.get_right_corners() != cv::Point(0, 0);
    if (side == LEFT_ONLY)
        return left_found;
    if (side == RIGHT_ONLY)
        return right_found;
    return left_found || right_found;
}

// 转弯判定状态机：拐点连续出现确认转弯开始，拐点连续消失确认转弯结束
struct CornerTurningState
{
    bool turning_active = false;
    bool turn_completed = false;
    int corner_detect_count = 0;
    int corner_missing_count = 0;
};

void updateCornerTurningState(CornerTurningState &s, bool corner_detected,
                              int corner_confirm_frames, int turning_end_confirm_frames)
{
    if (s.turn_completed)
        return;

    if (!s.turning_active)
    {
        s.corner_missing_count = 0;
        if (corner_detected)
        {
            if (s.corner_detect_count < corner_confirm_frames)
                ++s.corner_detect_count;
        }
        else
        {
            s.corner_detect_count = 0;
        }
        if (s.corner_detect_count >= corner_confirm_frames)
        {
            s.turning_active = true;
            s.corner_detect_count = 0;
            std::cout << "Corner confirmed for " << corner_confirm_frames
                      << " consecutive frames; turning started" << std::endl;
        }
        return;
    }

    s.corner_detect_count = 0;
    if (corner_detected)
    {
        s.corner_missing_count = 0;
    }
    else if (s.corner_missing_count < turning_end_confirm_frames)
    {
        ++s.corner_missing_count;
    }
    if (s.corner_missing_count >= turning_end_confirm_frames)
    {
        s.turning_active = false;
        s.turn_completed = true;
        s.corner_missing_count = 0;
        std::cout << "Corner absent for " << turning_end_confirm_frames
                  << " consecutive frames; turning finished" << std::endl;
    }
}

int main()
{

    // 参数
    int turning_end_x_error_abs_max_ = 15;                          // 转弯结束时 x_error 最大绝对值
    std::chrono::seconds corner_delay_s_ = std::chrono::seconds(3); // 直行状态延时进入CROSS状态的时间
    SearchSide straight_track_side = LEFT_ONLY;                     // STRAIGHT_TRACKING 走哪一边：LEFT_ONLY / RIGHT_ONLY / BOTH

    // 由 straight_track_side 一次性派生转弯前、转弯后的搜索侧与中线模式
    // （循环外计算，避免每帧 switch）
    SearchSide straight_side = straight_track_side;
    MidLineMode straight_mode = midLineModeForSide(straight_side);
    SearchSide post_turn_side = oppositeSide(straight_side);
    MidLineMode post_turn_mode = midLineModeForSide(post_turn_side);

    // 拐点确认与转弯结束确认的连续帧数（与 find_way_ros 默认值一致）
    int corner_confirm_frames = 3;
    int turning_end_confirm_frames = 3;
    CornerTurningState corner_turning;

    auto pre_t = std::chrono::steady_clock::now();
    auto cur_t = pre_t;
    long long sum_ms = 0;
    int frame_count = 0;
    float fps = 0.0;

    // 状态切换计时器
    std::chrono::_V2::steady_clock::time_point t0;

    // 转弯结束后可能出现的丢线情况，记录是否丢过线
    MissLineState miss_line = NO_MISS;

    // 状态信号
    InitSignal init_signal;
    init_signal.straight_signal = true; // 直接进入直行状态，测试用
    try
    {
        ProcessState state = ProcessState::IDLE;
        CameraCapture camera(0, 640, 480);
        ImageProcessConfig config;
        ImageProcess img_process(config);
        while (1)
        {
            cur_t = std::chrono::steady_clock::now();
            frame_count++;
            sum_ms += std::chrono::duration_cast<std::chrono::milliseconds>(cur_t - pre_t).count();
            if (frame_count % 30 == 0)
            {
                fps = 1000.0 / (sum_ms / 30.0);
                std::cout << "FPS: " << fps << std::endl;
                sum_ms = 0;
            }
            pre_t = cur_t;
            cv::Mat frame = camera.captureFrame();
            if (frame.empty())
            {
                std::cerr << "Error: Failed to capture frame from camera." << std::endl;
                continue;
            }
            cv::Mat frame_1 = camera.correctFrame(frame);
            if (frame_1.empty())
            {
                std::cerr << "Error: Frame correction failed, using original frame." << std::endl;
                frame_1 = frame;
            }
            cv::flip(frame_1, frame, 1); // 水平翻转

            if (state == ProcessState::IDLE)
            {
                if (init_signal.straight_signal)
                {
                    state = ProcessState::STRAIGHT_TRACKING;
                    t0 = std::chrono::steady_clock::now();
                    std::cout << "state: IDLE -> STRAIGHT_TRACKING" << std::endl;
                }
                else if (init_signal.left_signal)
                {
                    state = ProcessState::LEFT_TURNING;
                    std::cout << "state: IDLE -> LEFT_TURNING" << std::endl;
                }
                else if (init_signal.right_signal)
                {
                    state = ProcessState::RIGHT_TURNING;
                    std::cout << "state: IDLE -> RIGHT_TURNING" << std::endl;
                }
            }
            else
            {
                cv::Mat process_frame = frame;
                if (state == ProcessState::STRAIGHT_TRACKING)
                {
                    // 固定透视矩阵要求输入尺寸为 320x240；这一步不能交给
                    // preprocess，否则必须先完成透视变换才能得到处理图像。
                    cv::Mat perspective_input = frame;
                    img_process.resize_frame(perspective_input);
                    process_frame = vision_line::warpFixedGroundPerspective(perspective_input);
                }
                img_process.set_frame(process_frame);

                // preprocess 负责按配置缩放、有效区域背景估计、二值化和闭运算。
                cv::Mat binary_img = img_process.preprocess(process_frame);

                if (state == ProcessState::RIGHT_TURNING || state == ProcessState::LEFT_TURNING)
                {
                    cv::Mat canvas = img_process.return_frame();

                    img_process.get_side_line_task_1(binary_img, canvas, true);
                    img_process.fit_polynomial2();

                    img_process.draw_line(canvas, fps, state_name(state));

                    cv::imshow("binary", binary_img);
                    cv::imshow("canvas", canvas);
                }

                else
                {
                    // STRAIGHT_TRACKING
                    cv::Mat canvas = img_process.return_frame();

                    if (corner_turning.turn_completed)
                    {
                        // 转弯结束后切到相反单边，并采用与该侧匹配的偏移中线模式。
                        img_process.set_mid_line_mode(post_turn_mode);
                        img_process.get_side_line_task_2(binary_img, canvas, true, false, post_turn_side);
                    }
                    else
                    {
                        img_process.set_mid_line_mode(straight_mode);
                        img_process.get_side_line_task_2(binary_img, canvas, true, true, straight_side);
                        updateCornerTurningState(corner_turning, cornerDetected(img_process, straight_side),
                                                 corner_confirm_frames, turning_end_confirm_frames);

                        // 判定结束的这一帧已经按单边搜索过，立即重跑相反侧检测，
                        // 避免中线切换延迟到下一帧。
                        if (corner_turning.turn_completed)
                        {
                            img_process.set_mid_line_mode(post_turn_mode);
                            img_process.get_side_line_task_2(binary_img, canvas, true, false, post_turn_side);
                        }
                    }

                    // 本节点只在顶部按斜率向上补线，不向下补齐边线。
                    img_process.calculate_mid_line(binary_img, false);
                    img_process.fit_polynomial();

                    img_process.draw_line(canvas, fps, state_name(state));

                    cv::imshow("perspective", process_frame);
                    cv::imshow("binary", binary_img);
                    cv::imshow("canvas", canvas);
                }
            }

            cv::waitKey(1);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
