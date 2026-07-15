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
        CameraCapture camera(0, 1920, 1080); // Initialize camera with index 0 and resolution 1920x1080
        cv::Mat frame;
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
            }

            frame = camera.captureFrame();
            if (!frame.empty())
            {
                cv::imshow("Captured Frame", frame);
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