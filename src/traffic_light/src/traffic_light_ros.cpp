#include <chrono>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include "rknn_pool.hpp"
#include "yolov8_model.hpp"

namespace
{
const int kThreadCount = 3;
const char *kDefaultImageTopic = "/ucar_camera/image_raw";
const char *kWindowName = "Traffic Light Detection";

class TrafficLightRosNode
{
public:
    TrafficLightRosNode(const std::string &model_path,
                        const std::string &image_topic)
        : pool_(model_path, kThreadCount),
          image_topic_(image_topic),
          submitted_frames_(0),
          last_output_time_(Clock::now() - std::chrono::seconds(1))
    {
    }

    bool init()
    {
        if (pool_.init() != 0)
        {
            ROS_ERROR("rknnPool init failed");
            return false;
        }

        cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
        image_subscriber_ = node_handle_.subscribe(
            image_topic_, 1, &TrafficLightRosNode::imageCallback, this);
        ROS_INFO("Subscribed image topic: %s", image_topic_.c_str());
        return true;
    }

    ~TrafficLightRosNode()
    {
        cv::destroyWindow(kWindowName);
    }

private:
    using Clock = std::chrono::steady_clock;

    cv::Mat correctFrame(const cv::Mat &frame)
    {
        if (map_size_ != frame.size())
        {
            const cv::Mat camera_matrix =
                (cv::Mat_<double>(3, 3) << 420.88617453, 0.0, 322.17160714,
                 0.0, 423.22330218, 231.10564846,
                 0.0, 0.0, 1.0);
            const cv::Mat distortion =
                (cv::Mat_<double>(1, 5) << -0.34912917, 0.17532058,
                 0.01055267, -0.00249659, 0.0);
            const cv::Mat new_camera_matrix = cv::getOptimalNewCameraMatrix(
                camera_matrix, distortion, frame.size(), 0, frame.size());
            cv::initUndistortRectifyMap(
                camera_matrix, distortion, cv::Mat(), new_camera_matrix,
                frame.size(), CV_32FC1, map_x_, map_y_);
            map_size_ = frame.size();
        }

        cv::Mat corrected;
        cv::remap(frame, corrected, map_x_, map_y_, cv::INTER_LINEAR);
        return corrected;
    }

    void imageCallback(const sensor_msgs::ImageConstPtr &message)
    {
        cv::Mat frame;
        try
        {
            const cv_bridge::CvImageConstPtr cv_image = cv_bridge::toCvShare(
                message, sensor_msgs::image_encodings::BGR8);
            frame = correctFrame(cv_image->image);
            cv::flip(frame, frame, 1);
        }
        catch (const cv_bridge::Exception &error)
        {
            ROS_ERROR_THROTTLE(1.0, "cv_bridge conversion failed: %s", error.what());
            return;
        }
        catch (const cv::Exception &error)
        {
            ROS_ERROR_THROTTLE(1.0, "image preprocessing failed: %s", error.what());
            return;
        }

        if (frame.empty() || pool_.put(frame) != 0)
        {
            ROS_WARN_THROTTLE(1.0, "empty image or inference queue failure");
            return;
        }

        if (submitted_frames_ >= kThreadCount)
        {
            YoloV8Result result;
            if (pool_.get(result) != 0)
            {
                ROS_WARN_THROTTLE(1.0, "failed to get inference result");
                return;
            }

            const Clock::time_point now = Clock::now();
            if (now - last_output_time_ >= std::chrono::seconds(1))
            {
                showResult(result);
                last_output_time_ = now;
            }
        }

        ++submitted_frames_;
        if (cv::waitKey(1) == 'q')
        {
            ros::shutdown();
        }
    }

    void showResult(const YoloV8Result &result)
    {
        if (result.detections.empty())
        {
            ROS_INFO("No traffic light detected");
        }
        for (const TrafficLightDetection &detection : result.detections)
        {
            ROS_INFO("class=%s confidence=%.3f box=(%d, %d, %d, %d)",
                     detection.label.c_str(), detection.confidence,
                     detection.box.x, detection.box.y,
                     detection.box.x + detection.box.width,
                     detection.box.y + detection.box.height);
        }

        if (!result.image.empty())
        {
            cv::imshow(kWindowName, result.image);
        }
    }

    ros::NodeHandle node_handle_;
    ros::Subscriber image_subscriber_;
    rknnPool<YoloV8Model, cv::Mat, YoloV8Result> pool_;
    std::string image_topic_;
    long long submitted_frames_;
    Clock::time_point last_output_time_;
    cv::Mat map_x_;
    cv::Mat map_y_;
    cv::Size map_size_;
};
} // namespace

int main(int argc, char **argv)
{
    ros::init(argc, argv, "traffic_light_ros");
    ros::NodeHandle private_node_handle("~");

    std::string image_topic;
    private_node_handle.param<std::string>(
        "image_topic", image_topic, kDefaultImageTopic);

    const std::string package_path = ros::package::getPath("traffic_light");
    if (package_path.empty())
    {
        ROS_ERROR("Could not locate traffic_light package");
        return -1;
    }
    std::string model_path;
    private_node_handle.param<std::string>(
        "model_path", model_path, package_path + "/models/bestfp.rknn");

    TrafficLightRosNode node(model_path, image_topic);
    if (!node.init())
    {
        return -1;
    }

    ros::spin();
    return 0;
}
