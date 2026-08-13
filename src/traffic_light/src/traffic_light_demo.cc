#include <chrono>
#include <iostream>
#include <stdexcept>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "camera_capture.h"
#include "rknn_pool.hpp"
#include "yolov8_model.hpp"

static const char *kModelPath = "src/traffic_light/models/bestfp.rknn";
static const int kThreadCount = 3;
static const int kCameraIndex = 0;
static const int kCameraWidth = 640;
static const int kCameraHeight = 480;

int main()
{
    rknnPool<YoloV8Model, cv::Mat, YoloV8Result> pool(kModelPath, kThreadCount);
    if (pool.init() != 0)
    {
        std::cerr << "rknnPool init failed" << std::endl;
        return -1;
    }

    CameraCapture *camera = nullptr;
    try
    {
        camera = new CameraCapture(kCameraIndex, kCameraWidth, kCameraHeight);
    }
    catch (const std::exception &e)
    {
        std::cerr << "camera open failed: " << e.what() << std::endl;
        return -1;
    }

    cv::namedWindow("Camera FPS", cv::WINDOW_AUTOSIZE);

    int frames = 0;
    bool stop = false;
    using Clock = std::chrono::steady_clock;
    auto fps_window_start = Clock::now();
    int displayed_frames = 0;
    double current_fps = 0.0;

    auto show_result = [&](cv::Mat &result) {
        ++displayed_frames;
        const auto now = Clock::now();
        const double elapsed =
            std::chrono::duration<double>(now - fps_window_start).count();
        if (elapsed >= 1.0)
        {
            current_fps = displayed_frames / elapsed;
            displayed_frames = 0;
            fps_window_start = now;
        }

        const std::string fps_text = cv::format("FPS: %.2f", current_fps);
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
        cv::imshow("Camera FPS", result);
    };

    while (!stop)
    {
        cv::Mat frame;
        try
        {
            frame = camera->captureFrame();
            camera->correctFrame(frame);
            cv::flip(frame, frame, 1);
        }
        catch (const std::exception &e)
        {
            std::cerr << "camera capture failed: " << e.what() << std::endl;
            break;
        }

        if (frame.empty() || pool.put(frame) != 0)
        {
            break;
        }

        if (frames >= kThreadCount)
        {
            YoloV8Result result;
            if (pool.get(result) != 0)
            {
                break;
            }
            show_result(result.image);
        }

        if (cv::waitKey(1) == 'q')
        {
            stop = true;
        }
        ++frames;
    }

    while (!stop)
    {
        YoloV8Result result;
        if (pool.get(result) != 0)
        {
            break;
        }
        show_result(result.image);
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    delete camera;
    cv::destroyAllWindows();
    return 0;
}
