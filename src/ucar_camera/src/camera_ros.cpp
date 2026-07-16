#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <stdexcept>
#include <string>

#include "ucar_camera.h"

namespace
{
class CameraRosNode
{
public:
    CameraRosNode()
    {
        int camera_index = pnh_.param<int>("camera_index", 0);
        int image_width = pnh_.param<int>("image_width", 640);
        int image_height = pnh_.param<int>("image_height", 480);
        std::string topic = pnh_.param<std::string>("cam_topic_name", "/ucar_camera/image_raw");
        frame_id_ = pnh_.param<std::string>("frame_id", "opencv");
        int rate = pnh_.param<int>("rate", 30);
        enable_undistort_ = pnh_.param<bool>("enable_undistort", true);
        rate_hz_ = static_cast<double>(rate);

        cam_ = std::make_unique<CameraCapture>(camera_index, image_width, image_height);
        img_pub_ = nh_.advertise<sensor_msgs::Image>(topic, 1);

        ROS_INFO("camera_ros started: index=%d %dx%d topic=%s rate=%.1f undistort=%d",
                 camera_index, image_width, image_height, topic.c_str(), rate_hz_, enable_undistort_ ? 1 : 0);
    }

    void run()
    {
        ros::Rate rate(rate_hz_);
        sensor_msgs::Image img_msg;
        while (ros::ok())
        {
            cv::Mat frame = cam_->captureFrame();
            if (frame.empty())
            {
                ROS_WARN_THROTTLE(1.0, "camera capture returned empty frame");
                rate.sleep();
                continue;
            }
            if (enable_undistort_)
            {
                frame = cam_->correctFrame(frame);
            }
            cv::flip(frame, frame, 1);

            img_msg.header.stamp = ros::Time::now();
            img_msg.header.frame_id = frame_id_;
            img_msg.height = frame.rows;
            img_msg.width = frame.cols;
            img_msg.encoding = "bgr8";
            img_msg.is_bigendian = 0;
            img_msg.step = frame.cols * 3;
            img_msg.data.assign(frame.datastart, frame.dataend);

            img_pub_.publish(img_msg);
            rate.sleep();
        }
    }

private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_{"~"};
    ros::Publisher img_pub_;
    std::unique_ptr<CameraCapture> cam_;
    std::string frame_id_;
    bool enable_undistort_ = true;
    double rate_hz_ = 30.0;
};
}  // namespace

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ucar_camera");
    try
    {
        CameraRosNode node;
        node.run();
    }
    catch (const std::exception &e)
    {
        ROS_ERROR("ucar_camera node failed: %s", e.what());
        return 1;
    }
    return 0;
}
