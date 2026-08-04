#include <cerrno>
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

bool show_result(const cv::Mat &result)
{
    if (!result.empty())
        cv::imshow(kWindowName, result);
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

    rknnPool<PPOCRModel, cv::Mat, cv::Mat> pool(
        config.det_model_path, config.rec_model_path, config.thread_count);
    if (pool.init() != 0)
    {
        std::cerr << "PPOCR pool init failed" << std::endl;
        return -1;
    }

    int pending_count = 0;
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

            if (pool.put(frame) != 0)
            {
                std::cerr << "submit frame to PPOCR pool failed" << std::endl;
                break;
            }
            ++pending_count;

            if (pending_count >= config.thread_count)
            {
                cv::Mat result;
                if (pool.get(result) != 0)
                {
                    std::cerr << "get PPOCR result failed" << std::endl;
                    break;
                }
                --pending_count;
                stop = show_result(result);
            }
            else if ((cv::waitKey(1) & 0xff) == 'q')
            {
                stop = true;
            }
        }

        while (pending_count > 0)
        {
            cv::Mat result;
            if (pool.get(result) != 0)
            {
                std::cerr << "get pending PPOCR result failed" << std::endl;
                break;
            }
            --pending_count;
            show_result(result);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "find_signal_demo failed: " << e.what() << std::endl;
    }

    cv::destroyAllWindows();
    return pending_count == 0 ? 0 : -1;
}
