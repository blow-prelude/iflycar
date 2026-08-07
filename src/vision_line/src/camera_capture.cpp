#include "camera_capture.h"

namespace
{
    bool open_camera(cv::VideoCapture &capture, int camera_index, int width, int height,
                     bool convert_rgb)
    {
        if (!capture.open(camera_index, cv::CAP_V4L2))
        {
            return false;
        }

        capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
        capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        capture.set(cv::CAP_PROP_FPS, 30);
        capture.set(cv::CAP_PROP_BUFFERSIZE, 3);
        capture.set(cv::CAP_PROP_CONVERT_RGB, convert_rgb ? 1 : 0);
        return true;
    }
} // namespace

cv::Mat CameraCapture::perspective_matrix = (cv::Mat_<double>(3, 3) << -0.493484, -1.133680, 215.265780,
                                             -0.052136, -2.137570, 309.269947,
                                             -0.000221, -0.007728, 1.000000);
CameraCapture::CameraCapture()
{

    cap = new cv::VideoCapture();
    if (!open_camera(*cap, camera_index, frame_width, frame_height, true))
    {
        std::cerr << "Error: Could not open camera with index " << camera_index << std::endl;
        throw std::runtime_error("Could not open camera");
    }

    // frame = cv::Mat(frame_height, frame_width, CV_8UC3);
    std::cout << "Camera opened successfully with index " << camera_index << " and resolution " << frame_width << "x" << frame_height << std::endl;
}

CameraCapture::CameraCapture(int index, int width, int height) : camera_index(index), frame_width(width), frame_height(height)
{
    cap = new cv::VideoCapture();
    if (!open_camera(*cap, camera_index, frame_width, frame_height, true))
    {
        std::cerr << "Error: Could not open camera with index " << camera_index << std::endl;
        throw std::runtime_error("Could not open camera");
    }

    // frame = cv::Mat(frame_height, frame_width, CV_8UC3);
    std::cout << "Camera opened successfully with index " << camera_index << " and resolution " << frame_width << "x" << frame_height << std::endl;
}

CameraCapture::CameraCapture(int index, int width, int height, cv::Mat mtx, cv::Mat dist) : camera_index(index), frame_width(width), frame_height(height), mtx(mtx), dist(dist)
{
    cap = new cv::VideoCapture();
    if (!open_camera(*cap, camera_index, frame_width, frame_height, true))
    {
        std::cerr << "Error: Could not open camera with index " << camera_index << std::endl;
        throw std::runtime_error("Could not open camera");
    }

    // frame = cv::Mat(frame_height, frame_width, CV_8UC3);
    std::cout << "Camera opened successfully with index " << camera_index << " and resolution " << frame_width << "x" << frame_height << std::endl;
}

CameraCapture::CameraCapture(int index, int width, int height, cv::Mat mtx, cv::Mat dist, bool convert_rgb) : camera_index(index), frame_width(width), frame_height(height), mtx(mtx), dist(dist)
{
    cap = new cv::VideoCapture();
    if (!open_camera(*cap, camera_index, frame_width, frame_height, convert_rgb))
    {
        std::cerr << "Error: Could not open camera with index " << camera_index << std::endl;
        throw std::runtime_error("Could not open camera");
    }

    // frame = cv::Mat(frame_height, frame_width, CV_8UC3);
    std::cout << "Camera opened successfully with index " << camera_index << " and resolution " << frame_width << "x" << frame_height << std::endl;
}

CameraCapture::~CameraCapture()
{
    closeCamera();
    std::cout << "CameraCapture object destroyed, camera released." << std::endl;
}

cv::Mat CameraCapture::captureFrame()
{
    if (!cap->isOpened())
    {
        throw std::runtime_error("Camera is not opened");
    }

    cv::Mat frame;
    *cap >> frame;
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

cv::Mat CameraCapture::perspectiveFrame(const cv::Mat &frame)
{
    try
    {
        if (perspective_matrix.empty())
        {
            throw std::invalid_argument("Perspective matrix is empty");
        }

        cv::Mat dst;
        cv::warpPerspective(frame, dst, perspective_matrix, cv::Size(160, 144), cv::INTER_LINEAR);

        return dst;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error in perspectiveFrame: " << e.what() << std::endl;
        return frame;
    }
}

void CameraCapture::closeCamera()
{
    if (cap->isOpened())
    {
        cap->release();
    }
}
