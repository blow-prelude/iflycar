#include "camera_capture.h"
#include "ground_perspective.h"

#include <opencv2/highgui.hpp>

#include <exception>
#include <iostream>
#include <vector>

namespace
{
constexpr int kCameraIndex = 0;
constexpr int kCameraWidth = 640;
constexpr int kCameraHeight = 480;
constexpr int kFrameWidth = 320;
constexpr int kFrameHeight = 240;
constexpr int kGroundStartY = 139;
constexpr double kRealWidthM = 0.5;
constexpr double kRealLengthM = 0.75;
constexpr double kPixelsPerM = 200.0;

const std::vector<cv::Point2f> kSourcePoints{
    {114.0F, 146.0F}, {206.0F, 146.0F},
    {271.0F, 184.0F}, {35.0F, 187.0F}};
} // namespace

int main()
{
    try
    {
        CameraCapture camera(kCameraIndex, kCameraWidth, kCameraHeight);
        std::cout << "Camera stream started. Press q or Esc to exit."
                  << std::endl;
        while (true)
        {
            const cv::Mat raw_frame = camera.captureFrame();
            if (raw_frame.empty())
            {
                std::cerr << "Error: captured frame is empty." << std::endl;
                break;
            }
            const cv::Mat corrected_frame = camera.correctFrame(raw_frame);
            const cv::Mat frame = vision_line::flipAndResize(
                corrected_frame, {kFrameWidth, kFrameHeight});
            const cv::Mat ground_mask = vision_line::makeFixedGroundMask(
                frame.size(), kGroundStartY);
            const cv::Mat bird_view = vision_line::warpGround(
                frame, ground_mask, kSourcePoints,
                kRealWidthM, kRealLengthM, kPixelsPerM);

            cv::imshow("Camera Frame", frame);
            cv::imshow("Bird's Eye View", bird_view);
            cv::imshow("Ground Mask", ground_mask);
            if (vision_line::shouldExit(cv::waitKey(1)))
            {
                break;
            }
        }
        camera.closeCamera();
        cv::destroyAllWindows();
        return 0;
    }
    catch (const std::exception &error)
    {
        cv::destroyAllWindows();
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
}
