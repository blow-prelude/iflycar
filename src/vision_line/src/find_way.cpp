#include <opencv2/opencv.hpp>
#include "camera_capture.h"
#include "image_process.h"
#include <chrono>
#include <thread>

enum ProcessState
{
    IDLE = 0,
    STRAIGHT_TRACKING = 1,
    RIGHT_TURNING = 2,
    LEFT_TURNING = 3,
    CORNERING = 4,
    CROSS = 5,
    TURNING = 6,
    TRACKING2 = 7
};

struct InitSignal
{
    bool straight_signal = false;
    bool left_signal = false;
    bool right_signal = false;
};

int track_target_p(const std::vector<cv::Point> &line_points, int target_index, int w, int h)
{
    if (line_points.empty() || w <= 0 || h <= 0 || static_cast<int>(line_points.size()) < std::abs(target_index) + 1)
    {
        std::cout << "Invalid input for track_target_p" << std::endl;
        return -1000;
    }
    // 处理负索引
    int idx = target_index;
    if (idx < 0)
    {
        idx = static_cast<int>(line_points.size()) + idx;
    }
    if (idx < 0 || idx >= static_cast<int>(line_points.size()))
    {
        std::cout << "Index out of bounds for track_target_p" << std::endl;
        return -1000;
    }
    cv::Point pt = line_points[idx];
    double x_raw = pt.x;
    double y_raw = pt.y;
    double x_error = x_raw - (w / 2.0);
    std::cout << "Tracking target point: target idx:" << target_index << ", y:" << y_raw << ",x_error = " << x_error << std::endl;
    return x_error;
}

int main()
{

    // 参数
    int turning_end_x_error_abs_max_ = 15; // 转弯结束时 x_error 最大绝对值

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
                // img_process.set_frame(frame);

                // 预处理
                cv::Mat binary_img = img_process.preprocess(frame);

                if (state == ProcessState::RIGHT_TURNING || state == ProcessState::LEFT_TURNING)
                {
                    cv::Mat canvas = img_process.return_frame();

                    img_process.get_side_line_task_1(binary_img, canvas, true);
                    img_process.fit_polynomial2();

                    img_process.draw_line(canvas, fps, std::to_string(state));

                    cv::imshow("binary", binary_img);
                    cv::imshow("canvas", canvas);
                }

                else
                {
                    if (state == ProcessState::STRAIGHT_TRACKING)
                    {
                        // 延时3s后进入CROSS状态
                        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(3))
                        {
                            std::cout << "state: STRAIGHT_TRACKING -> CORSS after 3s" << std::endl;
                            state = ProcessState::CROSS;
                        }
                    }

                    cv::Mat canvas = img_process.return_frame();
                    img_process.get_side_line_task_2(binary_img, canvas, true, false);

                    if (state == ProcessState::CROSS)
                    {
                        std::vector<int> stop_mid = img_process.get_stop_line(binary_img, canvas, true);

                        // 检查是否进入TURNING状态
                        if (img_process.judge_enter_turning(stop_mid, binary_img.rows, binary_img.cols))
                        {
                            state = ProcessState::TURNING;
                            std::cout << "state: CROSS -> TURNING at y=" << stop_mid[1] / binary_img.rows << std::endl;
                        }
                    }

                    if (state == ProcessState::TURNING)
                    {
                        if (img_process.judge_turing_end(binary_img.cols, binary_img.rows, miss_line))
                        {
                            int x_error = track_target_p(img_process.get_fit_mid_line(), config.tracking2_target_p_index, binary_img.cols, binary_img.rows);
                            if (std::abs(x_error) <= turning_end_x_error_abs_max_)
                            {
                                state = ProcessState::TRACKING2;
                                std::cout << "state: TURNING -> TRACKING2" << std::endl;
                            }
                        }
                    }

                    if (state == ProcessState::TRACKING2)
                    {
                    }

                    img_process.fit_polynomial();
                    img_process.draw_line(canvas, fps, std::to_string(state));

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