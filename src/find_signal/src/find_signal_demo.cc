#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "camera_capture.h"
#include "ppocr_model.hpp"
#include "rknn_pool.hpp"

namespace
{
    const char *kDefaultDetModelPath = "src/find_signal/models/ppocrv4_det.rknn";
    const char *kDefaultRecModelPath = "src/find_signal/models/ppocrv4_rec.rknn";
    const int kDefaultCameraIndex = 0;
    const int kDefaultCameraWidth = 640;
    const int kDefaultCameraHeight = 480;
    const int kDefaultThreadCount = 2;
    const char *kWindowName = "PPOCR Camera";
    const int kFpsFrameCount = 60;

    struct FpsCounter
    {
        using Clock = std::chrono::steady_clock;

        Clock::time_point start_time = Clock::now();
        int frame_count = 0;
        double fps = 0.0;

        void on_frame()
        {
            ++frame_count;
            if (frame_count == kFpsFrameCount)
            {
                const double elapsed = std::chrono::duration<double>(
                    Clock::now() - start_time).count();
                if (elapsed > 0.0)
                    fps = kFpsFrameCount / elapsed;

                frame_count = 0;
                start_time = Clock::now();
            }
        }
    };

    struct DemoConfig
    {
        std::string det_model_path = kDefaultDetModelPath;
        std::string rec_model_path = kDefaultRecModelPath;
        int camera_index = kDefaultCameraIndex;
        int thread_count = kDefaultThreadCount;
    };

    void print_usage(const char *program)
    {
        std::cerr << "usage: " << program
                  << " [det_model] [rec_model] [camera_index] [thread_count]"
                  << std::endl;
    }

    bool parse_int(const char *text, int *value)
    {
        if (text == NULL || value == NULL || *text == '\0')
            return false;

        errno = 0;
        char *end = NULL;
        long parsed = std::strtol(text, &end, 10);
        if (errno != 0 || end == text || *end != '\0' ||
            parsed < -2147483647L || parsed > 2147483647L)
        {
            return false;
        }
        *value = static_cast<int>(parsed);
        return true;
    }

    bool parse_config(int argc, char **argv, DemoConfig *config)
    {
        if (config == NULL || argc < 1 || argc > 5)
            return false;

        if (argc >= 2)
            config->det_model_path = argv[1];
        if (argc >= 3)
            config->rec_model_path = argv[2];
        if (argc >= 4 && !parse_int(argv[3], &config->camera_index))
            return false;
        if (argc >= 5 && !parse_int(argv[4], &config->thread_count))
            return false;

        return config->thread_count >= 1 && config->thread_count <= 3;
    }

    bool show_result(cv::Mat &result, FpsCounter *fps_counter)
    {
        if (!result.empty())
        {
            fps_counter->on_frame();

            const std::string fps_text = cv::format("FPS: %.2f", fps_counter->fps);
            int baseline = 0;
            const cv::Size text_size = cv::getTextSize(
                fps_text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
            const int margin = 10;
            const cv::Point text_origin(
                std::max(margin, result.cols - text_size.width - margin),
                text_size.height + margin);
            const cv::Rect background(
                text_origin.x - 6,
                text_origin.y - text_size.height - 6,
                text_size.width + 12,
                text_size.height + baseline + 12);

            cv::rectangle(result, background, cv::Scalar(0, 0, 0), cv::FILLED);
            cv::putText(result, fps_text, text_origin,
                        cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            cv::imshow(kWindowName, result);
        }
        return (cv::waitKey(1) & 0xff) == 'q';
    }
} // namespace

int main(int argc, char **argv)
{
    DemoConfig config;
    if (!parse_config(argc, argv, &config))
    {
        print_usage(argv[0]);
        return -1;
    }

    rknnPool<PPOCRModel, cv::Mat, PPOCRInferenceResult> pool(
        config.det_model_path, config.rec_model_path, config.thread_count);
    if (pool.init() != 0)
    {
        std::cerr << "PPOCR pool init failed" << std::endl;
        return -1;
    }

    int pending_count = 0;
    FpsCounter fps_counter;
    bool stop = false;
    try
    {
        CameraCapture camera(config.camera_index, kDefaultCameraWidth,
                             kDefaultCameraHeight);
        cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);

        while (!stop)
        {
            cv::Mat frame = camera.captureFrame();

            if (frame.empty())
            {
                std::cerr << "camera returned an empty frame" << std::endl;
                break;
            }

            frame = camera.correctFrame(frame);
            cv::flip(frame, frame, 1);

            if (pool.put(frame) != 0)
            {
                std::cerr << "submit frame to PPOCR pool failed" << std::endl;
                break;
            }
            ++pending_count;

            if (pending_count >= config.thread_count)
            {
                PPOCRInferenceResult result;
                if (pool.get(result) != 0)
                {
                    std::cerr << "get PPOCR result failed" << std::endl;
                    break;
                }
                --pending_count;
                stop = show_result(result.image, &fps_counter);
            }
            else if ((cv::waitKey(1) & 0xff) == 'q')
            {
                stop = true;
            }
        }

        while (pending_count > 0)
        {
            PPOCRInferenceResult result;
            if (pool.get(result) != 0)
            {
                std::cerr << "get pending PPOCR result failed" << std::endl;
                break;
            }
            --pending_count;
            show_result(result.image, &fps_counter);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "find_signal_demo failed: " << e.what() << std::endl;
    }

    cv::destroyAllWindows();
    return pending_count == 0 ? 0 : -1;
}
