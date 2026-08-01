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
    rknnPool<YoloV8Model, cv::Mat, cv::Mat> pool(kModelPath, kThreadCount);
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
    while (!stop)
    {
        cv::Mat frame;
        try
        {
            frame = camera->captureFrame();
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
            cv::Mat result;
            if (pool.get(result) != 0)
            {
                break;
            }
            cv::imshow("Camera FPS", result);
        }

        if (cv::waitKey(1) == 'q')
        {
            stop = true;
        }
        ++frames;
    }

    while (!stop)
    {
        cv::Mat result;
        if (pool.get(result) != 0)
        {
            break;
        }
        cv::imshow("Camera FPS", result);
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    delete camera;
    cv::destroyAllWindows();
    return 0;
}
