#include <camera_capture.h>

int main()
{
    // 计算fps
    double prev_t = cv::getTickCount();
    double curr_t;
    double dt;
    int frame_count = 0;
    double fps = 0.0;
    try
    {
        CameraCapture camera(0, 640, 480); // Initialize camera with index 0 and resolution 640x480

        cv::Mat frame;
        cv::Mat mask;
        while (1)
        {
            curr_t = cv::getTickCount();
            dt = (curr_t - prev_t) / cv::getTickFrequency();
            prev_t = curr_t;
            frame_count++;
            if (dt >= 1.0) // Update FPS every second
            {
                fps = frame_count / dt;
                std::cout << "FPS: " << fps << std::endl;
                frame_count = 0;
                dt = 0.0;
            }

            frame = camera.captureFrame();
            if (!frame.empty())
            {
                camera.correctFrame(frame); // Apply distortion correction
                cv::flip(frame, frame, 1);  // Flip the frame horizontally for a mirror effect
                cv::resize(frame, frame, cv::Size(320, 240));
                cv::imshow("corrected Frame", frame);
                frame = CameraCapture::perspectiveFrame(frame);
                cv::imshow("perspective Frame", frame);
                cv::waitKey(1);
            }
            else
            {
                std::cerr << "Error: Captured frame is empty." << std::endl;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}