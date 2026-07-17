#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

class CameraCapture
{
public:
    CameraCapture();
    CameraCapture(int index, int width, int height);
    CameraCapture(int index, int width, int height, cv::Mat mtx, cv::Mat dist);
    ~CameraCapture();

    void close_camera();
    cv::Mat captureFrame();
    cv::Mat correctFrame(const cv::Mat &frame);
    static cv::Mat perspectiveFrame(const cv::Mat &frame);

private:
    int camera_index = 0;
    int frame_width = 640;
    int frame_height = 480;
    cv::VideoCapture cap;
    cv::Mat frame;
    cv::Mat mtx = (cv::Mat_<double>(3, 3) << 420.88617453, 0.0, 322.17160714,
                   0.0, 423.22330218, 231.10564846,
                   0.0, 0.0, 1.0);
    cv::Mat dist = (cv::Mat_<double>(1, 5) << -0.34912917, 0.17532058, 0.01055267, -0.00249659, 0.0);
    static cv::Mat perspective_matrix;

    bool is_camera_opened()
    {
        return cap.isOpened();
    }
};
