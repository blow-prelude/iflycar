#include <chrono>
#include <deque>
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
#include "traffic_light_decision.hpp"
#include "yolov8_model.hpp"

namespace
{
    const int kThreadCount = 3;
    const int kStartupDiscardFrames = 2;   // 跳过开头2帧
    const std::size_t kVoteWindowSize = 8; // 方向投票窗口大小
    const std::size_t kVoteThreshold = 5;  // 需要5票才能确认方向
    const std::size_t kFallbackFrameLimit = 15;
    const std::size_t kFallbackCvStreak = 3;
    const char *kDefaultImageTopic = "/ucar_camera/image_raw";
    const char *kDirectionTopic = "/vision_line_direction";
    const char *kWindowName = "Traffic Light Detection";
    const double kRoiStart = 0.4; // ROI 起始比例(x、y 均取 0.4~0.6)
    const double kRoiSpan = 0.2;  // ROI 宽高比例
    const char *kCvConflictFramePath = "/tmp/traffic_light_cv_conflict.jpg";

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
              discard_results_(kStartupDiscardFrames),
              decision_(kVoteWindowSize,
                        kVoteThreshold,
                        kFallbackFrameLimit,
                        kFallbackCvStreak)
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
            decision_.reset();
            if (enabled_)
            {
                discard_results_ = queued_frames_;
                conflict_frame_saved_ = false;
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
                frame = cv::Mat(cv_image->image.clone());
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

            // 先裁剪中心 ROI(x、y 均取 0.4~0.6),再送入模型处理
            const cv::Size original_size = frame.size();
            const int roi_x = static_cast<int>(frame.cols * kRoiStart);
            const int roi_y = static_cast<int>(frame.rows * kRoiStart);
            const int roi_width = static_cast<int>(frame.cols * kRoiSpan);
            const int roi_height = static_cast<int>(frame.rows * kRoiSpan);
            frame = frame(cv::Rect(roi_x, roi_y, roi_width, roi_height)).clone();
            ROS_INFO_ONCE("ROI crop %dx%d -> %dx%d (letterbox 会缩放到模型输入尺寸)",
                          original_size.width, original_size.height,
                          frame.cols, frame.rows);

            if (frame.empty() || pool_.put(frame) != 0)
            {
                ROS_WARN_THROTTLE(1.0, "empty image or inference queue failure");
                return;
            }
            ++queued_frames_;
            pending_frames_.push_back(frame);

            if (queued_frames_ > kThreadCount)
            {
                YoloV8Result result;
                if (pool_.get(result) != 0)
                {
                    ROS_WARN_THROTTLE(1.0, "failed to get inference result");
                    return;
                }
                --queued_frames_;

                cv::Mat source_frame;
                bool has_source_frame = false;
                if (!pending_frames_.empty())
                {
                    source_frame = pending_frames_.front();
                    pending_frames_.pop_front();
                    has_source_frame = true;
                }
                else
                {
                    ROS_ERROR_THROTTLE(1.0,
                                       "inference result has no matching source frame");
                }

                if (discard_results_ > 0)
                {
                    --discard_results_;
                }
                else
                {
                    showResult(result);
                    if (has_source_frame)
                    {
                        updateDirectionDecision(result, source_frame);
                    }
                    else
                    {
                        ROS_WARN_THROTTLE(1.0,
                                          "skipping direction decision without source frame");
                    }
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

        void updateDirectionDecision(const YoloV8Result &result,
                                     const cv::Mat &source_frame)
        {
            const TrafficLightDetection *best_direction = bestDirection(result);
            std::string model_label;
            std::string cv_label = "unknown";
            if (best_direction != nullptr)
            {
                model_label = best_direction->label;
                if ((model_label == "left" || model_label == "right") &&
                    !source_frame.empty())
                {
                    try
                    {
                        const Clock::time_point cv_start = Clock::now();
                        cv_label = traffic_light::classifyArrowDirection(
                            source_frame, best_direction->box);
                        const long long cv_elapsed_us =
                            std::chrono::duration_cast<std::chrono::microseconds>(
                                Clock::now() - cv_start)
                                .count();
                        ROS_INFO("CV arrow: model=%s cv=%s confidence=%.3f "
                                 "frame=%dx%d box=(%d,%d,%d,%d) elapsed=%lldus "
                                 "rule=green-bright-component+pca+wing-offset",
                                 model_label.c_str(), cv_label.c_str(),
                                 best_direction->confidence,
                                 source_frame.cols, source_frame.rows,
                                 best_direction->box.x, best_direction->box.y,
                                 best_direction->box.x + best_direction->box.width,
                                 best_direction->box.y + best_direction->box.height,
                                 cv_elapsed_us);
                    }
                    catch (const cv::Exception &error)
                    {
                        ROS_WARN_THROTTLE(1.0,
                                          "traffic-light CV classification failed: %s",
                                          error.what());
                    }
                }
            }

            const bool was_fallback = decision_.inFallback();
            const bool model_cv_conflict =
                (model_label == "left" || model_label == "right") &&
                (cv_label == "left" || cv_label == "right") &&
                model_label != cv_label;
            if (model_cv_conflict && !conflict_frame_saved_)
            {
                conflict_frame_saved_ = true;
                try
                {
                    if (cv::imwrite(kCvConflictFramePath, source_frame))
                    {
                        ROS_WARN("Saved model/CV conflict frame to %s: "
                                 "model=%s cv=%s box=(%d, %d, %d, %d)",
                                 kCvConflictFramePath,
                                 model_label.c_str(),
                                 cv_label.c_str(),
                                 best_direction->box.x,
                                 best_direction->box.y,
                                 best_direction->box.x + best_direction->box.width,
                                 best_direction->box.y + best_direction->box.height);
                    }
                    else
                    {
                        ROS_WARN("Failed to save model/CV conflict frame to %s",
                                 kCvConflictFramePath);
                    }
                }
                catch (const cv::Exception &error)
                {
                    ROS_WARN("Failed to save model/CV conflict frame: %s",
                             error.what());
                }
            }

            const std::size_t next_frame = decision_.processedFrameCount() + 1;
            const bool inference_enters_window =
                !was_fallback && (model_label == "straight" ||
                                  model_label == "left" ||
                                  model_label == "right") &&
                !model_cv_conflict;
            if (inference_enters_window)
            {
                ROS_INFO("vote-window input: frame=%zu source=inference label=%s",
                         next_frame, model_label.c_str());
            }

            const bool cv_enters_window =
                inference_enters_window &&
                (model_label == "left" || model_label == "right") &&
                model_label == cv_label;
            if (cv_enters_window)
            {
                ROS_INFO("vote-window input: frame=%zu source=cv-processing label=%s",
                         next_frame, cv_label.c_str());
            }
            if (!was_fallback && model_cv_conflict)
            {
                ROS_INFO("vote-window skipped: frame=%zu "
                         "source=inference label=%s "
                         "source=cv-processing label=%s reason=conflict",
                         next_frame,
                         model_label.c_str(),
                         cv_label.c_str());
            }

            const std::string final_label = decision_.processFrame(
                model_label, cv_label);
            if (!was_fallback && decision_.inFallback())
            {
                ROS_WARN("No traffic-light vote reached %zu votes in %zu frames; "
                         "switching to CV-only fallback",
                         kVoteThreshold, kFallbackFrameLimit);
            }

            ROS_INFO_THROTTLE(
                1.0,
                "direction frame=%zu model=%s cv=%s votes=%zu "
                "straight=%d left=%d right=%d mode=%s cv_candidate=%s cv_streak=%zu",
                decision_.processedFrameCount(),
                model_label.empty() ? "unknown" : model_label.c_str(),
                cv_label.c_str(),
                decision_.voteCount(),
                decision_.voteCountFor("straight"),
                decision_.voteCountFor("left"),
                decision_.voteCountFor("right"),
                decision_.inFallback() ? "cv-only" : "vote",
                decision_.cvCandidate().empty() ? "none" : decision_.cvCandidate().c_str(),
                decision_.cvStreak());

            if (final_label.empty())
            {
                return;
            }

            std_msgs::String message;
            message.data = final_label;
            direction_publisher_.publish(message);
            ROS_INFO("Published traffic-light direction: %s (source=%s); waiting silently",
                     message.data.c_str(),
                     was_fallback ? "cv-fallback" : "vote");

            decision_.reset();
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
        std::deque<cv::Mat> pending_frames_;
        traffic_light::DirectionDecisionAccumulator decision_;
        bool window_created_ = false;
        bool conflict_frame_saved_ = false;
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
