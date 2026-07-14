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

CameraCapture::CameraCapture(int index, int width, int height, cv::Mat mtx, cv::Mat dist) : camera_index(index), frame_width(width), frame_height(height), mtx(mtx), dist(dist)
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

cv::Mat CameraCapture::correctFrame(const cv::Mat &frame)
{
    try
    {
        if (this->mtx.empty() || this->dist.empty())
        {
            throw std::invalid_argument("Camera matrix or distortion coefficients are empty");
        }

        int h = frame.rows;
        int w = frame.cols;
        // 计算新的内参矩阵
        cv::Rect roi;
        cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(this->mtx, this->dist, cv::Size(w, h), 0, cv::Size(w, h), &roi);

        // 生成去畸变表
        cv::Mat mapx, mapy;
        cv::initUndistortRectifyMap(this->mtx, this->dist, cv::Mat(), newCameraMatrix, cv::Size(w, h), CV_32FC1, mapx, mapy);

        // 应用去畸变映射
        cv::Mat dst;
        cv::remap(frame, dst, mapx, mapy, cv::INTER_LINEAR);

        return dst;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error in correctFrame: " << e.what() << std::endl;
        return frame; // 返回原始帧
    }
}

void CameraCapture::close_camera()
{
    if (cap.isOpened())
    {
        cap.release();
    }
}
