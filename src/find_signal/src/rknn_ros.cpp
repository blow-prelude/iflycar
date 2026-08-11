#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <initializer_list>
#include <queue>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32.h>
#include <std_srvs/SetBool.h>

#include "ppocr_model.hpp"
#include "rknn_pool.hpp"

int classfy(const std::string &text)
{
    auto contains = [&text](std::initializer_list<const char *> keywords)
    {
        for (const char *kw : keywords)
        {
            if (text.find(kw) != std::string::npos)
            {
                return true;
            }
        }
        return false;
    };

    if (contains({"食品", "食"}))
    {
        return 0;
    }
    if (contains({"日用品", "日", "用品"}))
    {
        return 1;
    }
    if (contains({"电子产品", "电子", "电", "生产"}))
    {
        return 2;
    }
    return -1;
}

namespace
{
    const char *kImageTopic = "/ucar_camera/image_raw";
    const char *kDetectionTopic = "/signal_detection";
    const char *kClassTopic = "/signal_class";
    const char *kWindowName = "Signal Detection Result";
    const int kMaxThreadCount = 3;

    struct NodeConfig
    {
        std::string det_model_path;
        std::string rec_model_path;
        int thread_count;
        bool visualize;
    };

    double box_area(const rknn_quad_t &box)
    {
        const rknn_point_t points[] = {
            box.left_top,
            box.right_top,
            box.right_bottom,
            box.left_bottom,
        };

        double twice_area = 0.0;
        for (int i = 0; i < 4; ++i)
        {
            const rknn_point_t &current = points[i];
            const rknn_point_t &next = points[(i + 1) % 4];
            twice_area += static_cast<double>(current.x) * next.y -
                          static_cast<double>(current.y) * next.x;
        }
        return std::abs(twice_area) * 0.5;
    }

    const ppocr_text_recog_result_t *get_biggest_result(
        const ppocr_text_recog_array_result_t &results)
    {
        const int result_count = std::max(0, std::min(results.count, 1000));
        const ppocr_text_recog_result_t *biggest = NULL;
        double biggest_area = -1.0;

        for (int i = 0; i < result_count; ++i)
        {
            const ppocr_text_recog_result_t &candidate = results.text_result[i];
            const double area = box_area(candidate.box);
            if (area > biggest_area)
            {
                biggest_area = area;
                biggest = &candidate;
            }
        }
        return biggest;
    }

    NodeConfig load_config(ros::NodeHandle &private_node)
    {
        NodeConfig config;
        const std::string package_path = ros::package::getPath("find_signal");
        config.det_model_path = package_path + "/models/ppocrv4_det.rknn";
        config.rec_model_path = package_path + "/models/ppocrv4_rec.rknn";
        config.thread_count = 3;
        config.visualize = true;

        private_node.param<std::string>("det_model_path", config.det_model_path,
                                        config.det_model_path);
        private_node.param<std::string>("rec_model_path", config.rec_model_path,
                                        config.rec_model_path);
        private_node.param("thread_count", config.thread_count, config.thread_count);
        private_node.param("visualize", config.visualize, config.visualize);
        return config;
    }

    class SignalDetectionNode
    {
    public:
        SignalDetectionNode(const NodeConfig &config)
            : pool_(config.det_model_path, config.rec_model_path,
                    config.thread_count),
              thread_count_(config.thread_count),
              pending_count_(0),
              visualize_(config.visualize),
              enabled_(false),
              processed_frame_count_(0)
        {
        }

        ~SignalDetectionNode()
        {
            if (visualize_)
                cv::destroyWindow(kWindowName);
        }

        bool init()
        {
            if (pool_.init() != 0)
            {
                ROS_ERROR("PPOCR pool initialization failed");
                return false;
            }

            detection_pub_ = node_.advertise<std_msgs::Float32MultiArray>(
                kDetectionTopic, 10);
            class_pub_ = node_.advertise<std_msgs::Int32>(kClassTopic, 10);
            image_sub_ = node_.subscribe(kImageTopic, 1,
                                         &SignalDetectionNode::image_callback, this);
            enable_service_ = private_node_.advertiseService(
                "set_enabled", &SignalDetectionNode::set_enabled_callback, this);
            if (visualize_)
                cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
            ROS_INFO("find_signal ready (OCR disabled): subscribe %s, publish %s and %s",
                     kImageTopic, kDetectionTopic, kClassTopic);
            return true;
        }

    private:
        void image_callback(const sensor_msgs::ImageConstPtr &message)
        {
            if (!enabled_)
                return;

            const std::chrono::steady_clock::time_point preprocess_start =
                std::chrono::steady_clock::now();
            cv_bridge::CvImageConstPtr cv_image;
            try
            {
                cv_image = cv_bridge::toCvShare(
                    message, sensor_msgs::image_encodings::BGR8);
            }
            catch (const cv_bridge::Exception &error)
            {
                ROS_ERROR_THROTTLE(1.0, "cv_bridge conversion failed: %s",
                                   error.what());
                return;
            }

            // The callback message may be released before a pool worker runs.
            // Clone the frame so the queued cv::Mat owns its image buffer.
            const cv::Mat frame = cv_image->image.clone();
            const double callback_preprocess_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - preprocess_start)
                    .count();
            if (pool_.put(frame) != 0)
            {
                ROS_ERROR_THROTTLE(1.0, "failed to submit an image to PPOCR pool");
                return;
            }
            callback_preprocess_times_.push(callback_preprocess_ms);
            ++pending_count_;

            // Keep at most one inference per NPU worker outstanding, matching the
            // pipelined behavior used by find_signal_demo.
            if (pending_count_ < thread_count_)
                return;

            PPOCRInferenceResult inference_result;
            try
            {
                if (pool_.get(inference_result) != 0)
                {
                    ROS_ERROR_THROTTLE(1.0, "failed to get a PPOCR result");
                    return;
                }
            }
            catch (const std::exception &error)
            {
                ROS_FATAL("PPOCR worker failed: %s", error.what());
                ros::shutdown();
                return;
            }
            --pending_count_;

            if (!callback_preprocess_times_.empty())
            {
                inference_result.timing.preprocess_ms +=
                    callback_preprocess_times_.front();
                callback_preprocess_times_.pop();
            }

            const std::chrono::steady_clock::time_point postprocess_start =
                std::chrono::steady_clock::now();
            show_result(inference_result.image);
            publish_result(inference_result.ocr_results);
            inference_result.timing.postprocess_ms +=
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - postprocess_start)
                    .count();

            ++processed_frame_count_;
            const double total_ms = inference_result.timing.preprocess_ms +
                                    inference_result.timing.inference_ms +
                                    inference_result.timing.postprocess_ms;
            ROS_INFO("frame %llu timing: preprocess=%.3f ms, "
                     "inference=%.3f ms, postprocess=%.3f ms, total=%.3f ms",
                     processed_frame_count_,
                     inference_result.timing.preprocess_ms,
                     inference_result.timing.inference_ms,
                     inference_result.timing.postprocess_ms,
                     total_ms);
        }

        bool set_enabled_callback(std_srvs::SetBool::Request &request,
                                  std_srvs::SetBool::Response &response)
        {
            if (enabled_ == request.data)
            {
                response.success = true;
                response.message = enabled_ ? "already enabled" : "already disabled";
                return true;
            }

            enabled_ = request.data;
            response.success = true;
            response.message = enabled_ ? "enabled" : "disabled";
            ROS_INFO("OCR processing %s", response.message.c_str());
            return true;
        }

        void show_result(const cv::Mat &image)
        {
            if (!visualize_ || image.empty())
                return;

            // PPOCRModel::infer has already drawn every recognized quadrangle.
            cv::imshow(kWindowName, image);
            cv::waitKey(1);
        }

        void publish_result(const ppocr_text_recog_array_result_t &results)
        {
            const ppocr_text_recog_result_t *result = get_biggest_result(results);
            if (result == NULL)
                return;

            const rknn_quad_t &box = result->box;
            std_msgs::Float32MultiArray detection_message;
            detection_message.data.resize(4);
            detection_message.data[0] =
                (box.left_top.x + box.right_top.x + box.right_bottom.x +
                 box.left_bottom.x) /
                4.0f;
            detection_message.data[1] =
                (box.left_top.y + box.right_top.y + box.right_bottom.y +
                 box.left_bottom.y) /
                4.0f;
            detection_message.data[2] =
                (box.left_top.x + box.left_bottom.x) / 2.0f;
            detection_message.data[3] =
                (box.right_top.x + box.right_bottom.x) / 2.0f;
            detection_pub_.publish(detection_message);

            const std::string text(result->text.str);
            const int class_id = classfy(text);
            ROS_INFO("OCR text: %s, class_id: %d", text.c_str(), class_id);
            if (class_id == -1)
                return;

            std_msgs::Int32 class_message;
            class_message.data = class_id;
            class_pub_.publish(class_message);
        }

        ros::NodeHandle node_;
        ros::NodeHandle private_node_{"~"};
        ros::Subscriber image_sub_;
        ros::ServiceServer enable_service_;
        ros::Publisher detection_pub_;
        ros::Publisher class_pub_;
        rknnPool<PPOCRModel, cv::Mat, PPOCRInferenceResult> pool_;
        std::queue<double> callback_preprocess_times_;
        int thread_count_;
        int pending_count_;
        bool visualize_;
        bool enabled_;
        unsigned long long processed_frame_count_;
    };
} // namespace

int main(int argc, char **argv)
{
    ros::init(argc, argv, "find_signal");
    ros::NodeHandle private_node("~");
    const NodeConfig config = load_config(private_node);

    if (config.thread_count < 1 || config.thread_count > kMaxThreadCount)
    {
        ROS_FATAL("~thread_count must be between 1 and %d", kMaxThreadCount);
        return 1;
    }

    SignalDetectionNode node(config);
    if (!node.init())
        return 1;

    ros::spin();
    return 0;
}
