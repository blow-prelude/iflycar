#include "camera_capture.h"

CameraCapture::CameraCapture()
{
    cap.open(camera_index);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera with index " << camera_index << std::endl;
        throw std::runtime_error("Could not open camera");
    }
    else
    {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, frame_width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height);

        frame = cv::Mat(frame_height, frame_width, CV_8UC3);
    }
}

CameraCapture::CameraCapture(int index, int width, int height) : camera_index(index), frame_width(width), frame_height(height)
{
    cap.open(camera_index);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera with index " << camera_index << std::endl;
        throw std::runtime_error("Could not open camera");
    }
    else
    {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, frame_width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height);
        frame = cv::Mat(frame_height, frame_width, CV_8UC3);
        std::cout << "Camera opened successfully with index " << camera_index << " and resolution " << frame_width << "x" << frame_height << std::endl;
    }
}

CameraCapture::~CameraCapture()
{
    close_camera();
    std::cout << "CameraCapture object destroyed, camera released." << std::endl;
}

cv::Mat CameraCapture::captureFrame()
{
    if (!cap.isOpened())
    {
        throw std::runtime_error("Camera is not opened");
    }

    cv::Mat frame;
    cap >> frame;
    return frame;
}

void CameraCapture::close_camera()
{
    if (cap.isOpened())
    {
        cap.release();
    }
}
