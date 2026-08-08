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

// 水平白线判定转弯结束的状态：初始即视为转弯进行中，只判定结束。
struct StopLineTurningState
{
    bool turn_completed = false; // 置位后常驻相反侧巡线，不再判定
    int stop_line_end_count = 0; // 中点 y 超过阈值的连续帧数
};

// 用水平白线判定转弯是否结束：中点归一化 y 超过 y_thresh，连续 confirm_frames 帧即结束。
// 每帧在 canvas 上绘制停止线中点与阈值参考线，便于标定 y_thresh。
// 返回 true 表示本次调用刚判定转弯结束（调用方需同帧切到相反侧巡线）。
bool checkStopLineTurnEnd(ImageProcess &img_process, cv::Mat &binary, cv::Mat &canvas,
                          float y_thresh, int confirm_frames, StopLineTurningState &state)
{
    if (state.turn_completed)
        return false;

    std::vector<int> stop_line = img_process.get_stop_line(binary, canvas, true);

    bool stop_line_low = false;
    if (!stop_line.empty() && binary.rows > 0)
    {
        const float y_norm = stop_line[1] / static_cast<float>(binary.rows);
        stop_line_low = y_norm > y_thresh;

        // 可视化：阈值参考线（绿）+ 中点高亮（黄环），便于观察中点何时越过阈值
        if (canvas.rows > 0)
        {
            const int thresh_y = static_cast<int>(y_thresh * canvas.rows);
            cv::line(canvas, cv::Point(0, thresh_y), cv::Point(canvas.cols, thresh_y),
                     cv::Scalar(0, 255, 0), 1);
            const cv::Point center(stop_line[0], stop_line[1]);
            cv::circle(canvas, center, 8, cv::Scalar(0, 255, 255), 2);
        }
    }

    if (stop_line_low)
    {
        if (state.stop_line_end_count < confirm_frames)
            ++state.stop_line_end_count;
    }
    else
    {
        state.stop_line_end_count = 0;
    }

    if (state.stop_line_end_count >= confirm_frames)
    {
        state.turn_completed = true;
        state.stop_line_end_count = 0;
        std::cout << "Stop line midpoint y_norm > " << y_thresh
                  << " for " << confirm_frames
                  << " consecutive frames; turning finished" << std::endl;
        return true;
    }
    return false;
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

    // 水平白线判定转弯结束：中点归一化 y 超过阈值，连续 turning_end_confirm_frames 帧即结束
    float stop_line_end_y_thresh = 0.55f;
    int turning_end_confirm_frames = 3;
    // 初始即视为转弯进行中，只判定结束；turn_completed 置位后常驻相反侧巡线
    StopLineTurningState stop_line_turning;

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

                    if (stop_line_turning.turn_completed)
                    {
                        // 转弯结束后常驻相反单边，并采用与该侧匹配的偏移中线模式。
                        img_process.set_mid_line_mode(post_turn_mode);
                        img_process.get_side_line_task_2(binary_img, canvas, true, false, post_turn_side);
                    }
                    else
                    {
                        // 转弯进行中：初始侧巡线，不再检测拐点（find_corner=false）。
                        img_process.set_mid_line_mode(straight_mode);
                        img_process.get_side_line_task_2(binary_img, canvas, true, false, straight_side);

                        // 用水平白线判定转弯是否结束；刚结束则同帧切到相反侧巡线，
                        // 避免中线切换延迟到下一帧。
                        if (checkStopLineTurnEnd(img_process, binary_img, canvas,
                                                 stop_line_end_y_thresh, turning_end_confirm_frames,
                                                 stop_line_turning))
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
