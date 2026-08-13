#include <chrono>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/String.h>
#include <std_srvs/SetBool.h>

#include "rknn_pool.hpp"
#include "yolov8_model.hpp"

namespace
{
    const int kThreadCount = 3;
    const int kRequiredConsecutiveFrames = 5;
    const char *kDefaultImageTopic = "/ucar_camera/image_raw";
    const char *kDirectionTopic = "/vision_line_direction";
    const char *kWindowName = "Traffic Light Detection";

    class TrafficLightRosNode
    {
    public:
        TrafficLightRosNode(const std::string &model_path,
                            const std::string &image_topic,
                            bool initial_enabled)
            : pool_(model_path, kThreadCount),
              image_topic_(image_topic),
              enabled_(initial_enabled),
              queued_frames_(0),
              discard_results_(0),
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

            if (enabled_)
            {
                createWindow();
            }
            image_subscriber_ = node_handle_.subscribe(
                image_topic_, 1, &TrafficLightRosNode::imageCallback, this);
            direction_publisher_ = node_handle_.advertise<std_msgs::String>(
                kDirectionTopic, 1);
            enable_service_ = private_node_handle_.advertiseService(
                "set_enabled", &TrafficLightRosNode::setEnabled, this);
            ROS_INFO("Subscribed image topic: %s", image_topic_.c_str());
            ROS_INFO("Publishing direction topic: %s", kDirectionTopic);
            ROS_INFO("Traffic-light processing initially %s",
                     enabled_ ? "enabled" : "disabled");
            return true;
        }

        ~TrafficLightRosNode()
        {
            if (window_created_)
            {
                cv::destroyWindow(kWindowName);
            }
        }

    private:
        using Clock = std::chrono::steady_clock;

        void createWindow()
        {
            if (!window_created_)
            {
                cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
                window_created_ = true;
            }
        }

        bool setEnabled(std_srvs::SetBool::Request &request,
                        std_srvs::SetBool::Response &response)
        {
            if (request.data == enabled_)
            {
                response.success = true;
                response.message = enabled_ ? "already enabled" : "already disabled";
                return true;
            }

            enabled_ = request.data;
            direction_candidate_.clear();
            direction_streak_ = 0;
            if (enabled_)
            {
                discard_results_ = queued_frames_;
                last_output_time_ = Clock::now() - std::chrono::seconds(1);
                createWindow();
            }
            else if (window_created_)
            {
                cv::destroyWindow(kWindowName);
                window_created_ = false;
            }

            response.success = true;
            response.message = enabled_ ? "enabled" : "disabled";
            ROS_INFO("Traffic-light processing %s", response.message.c_str());
            return true;
        }

        void imageCallback(const sensor_msgs::ImageConstPtr &message)
        {
            if (!enabled_)
            {
                return;
            }

            cv::Mat frame;
            try
            {
                const cv_bridge::CvImageConstPtr cv_image = cv_bridge::toCvShare(
                    message, sensor_msgs::image_encodings::BGR8);
                frame = cv::Mat(cv_image->image);
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
            ++queued_frames_;

            if (queued_frames_ > kThreadCount)
            {
                YoloV8Result result;
                if (pool_.get(result) != 0)
                {
                    ROS_WARN_THROTTLE(1.0, "failed to get inference result");
                    return;
                }
                --queued_frames_;

                if (discard_results_ > 0)
                {
                    --discard_results_;
                }
                else
                {
                    const Clock::time_point now = Clock::now();
                    if (now - last_output_time_ >= std::chrono::seconds(1))
                    {
                        showResult(result);
                        last_output_time_ = now;
                    }
                    updateDirectionStreak(result);
                }
            }

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

        const TrafficLightDetection *bestDirection(const YoloV8Result &result) const
        {
            const TrafficLightDetection *best_direction = nullptr;
            for (const TrafficLightDetection &detection : result.detections)
            {
                const bool publishable = detection.label == "straight" ||
                                         detection.label == "right" ||
                                         detection.label == "left";
                if (publishable &&
                    (best_direction == nullptr ||
                     detection.confidence > best_direction->confidence))
                {
                    best_direction = &detection;
                }
            }
            return best_direction;
        }

        void updateDirectionStreak(const YoloV8Result &result)
        {
            const TrafficLightDetection *best_direction = bestDirection(result);
            if (best_direction == nullptr)
            {
                direction_candidate_.clear();
                direction_streak_ = 0;
                return;
            }

            if (best_direction->label == direction_candidate_)
            {
                ++direction_streak_;
            }
            else
            {
                direction_candidate_ = best_direction->label;
                direction_streak_ = 1;
            }

            if (direction_streak_ < kRequiredConsecutiveFrames)
            {
                return;
            }

            std_msgs::String message;
            message.data = direction_candidate_;
            direction_publisher_.publish(message);
            ROS_INFO("Published traffic-light direction after %d consecutive frames: %s; waiting silently",
                     kRequiredConsecutiveFrames, message.data.c_str());

            direction_candidate_.clear();
            direction_streak_ = 0;
            enabled_ = false;
            if (window_created_)
            {
                cv::destroyWindow(kWindowName);
                window_created_ = false;
            }
        }

        ros::NodeHandle node_handle_;
        ros::NodeHandle private_node_handle_{"~"};
        ros::Subscriber image_subscriber_;
        ros::Publisher direction_publisher_;
        ros::ServiceServer enable_service_;
        rknnPool<YoloV8Model, cv::Mat, YoloV8Result> pool_;
        std::string image_topic_;
        bool enabled_;
        int queued_frames_;
        int discard_results_;
        std::string direction_candidate_;
        int direction_streak_ = 0;
        bool window_created_ = false;
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
    bool initial_enabled;
    private_node_handle.param("initial_enabled", initial_enabled, true);

    const std::string package_path = ros::package::getPath("traffic_light");
    if (package_path.empty())
    {
        ROS_ERROR("Could not locate traffic_light package");
        return -1;
    }
    std::string model_path;
    private_node_handle.param<std::string>(
        "model_path", model_path, package_path + "/models/bestfp.rknn");

    TrafficLightRosNode node(model_path, image_topic, initial_enabled);
    if (!node.init())
    {
        return -1;
    }

    ros::spin();
    return 0;
}
