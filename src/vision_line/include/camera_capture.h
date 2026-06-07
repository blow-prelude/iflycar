#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

class CameraCapture
{
public:
    CameraCapture();
    CameraCapture(int index, int width, int height);
    ~CameraCapture();

    void close_camera();
    cv::Mat captureFrame();

private:
    int camera_index = 0;
    int frame_width = 640;
    int frame_height = 480;
    cv::VideoCapture cap;
    cv::Mat frame;

    bool is_camera_opened()
    {
        return cap.isOpened();
    }
};
