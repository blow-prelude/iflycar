#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/String.h>
#include <opencv2/opencv.hpp>
#include <mutex>

#include "image_process.h"
#include "camera_capture.h"

enum State
{
    IDLE = 0,
    STRAIGHT_TRACKING = 1,
    RIGHT_TRACKING = 2,
    LEFT_TRACKING = 3,

};

const char *state_name(State s)
{
    switch (s)
    {
    case IDLE:
        return "IDLE";
    case STRAIGHT_TRACKING:
        return "STRAIGHT_TRACKING";
    case RIGHT_TRACKING:
        return "RIGHT_TRACKING";
    case LEFT_TRACKING:
        return "LEFT_TRACKING";
    default:
        return "UNKNOWN";
    }
}

class FindWayROS
{
public:
    FindWayROS() : nh_private_("~"), it_(nh_), state_(IDLE), miss_line_(NO_MISS)
    {
        // 输出初始配置值
        // ROS_INFO("Initial config values: straight_target_p_index=%d, left_target_p_index=%d, tracking2_target_p_index=%d",
        //          config_.straight_target_p_index, config_.left_target_p_index, config_.tracking2_target_p_index);

        // 加载 ROS 参数（私有命名空间）
        std::string image_topic = nh_private_.param<std::string>("image_topic", "ucar_camera/image_raw");
        std::string vision_line_topic = nh_private_.param<std::string>("vision_line_topic", "/vision_line");
        turning_flag_param_ = nh_private_.param<std::string>("turning_flag_param", "/start_vision_line2");
        direction_topic_ = nh_private_.param<std::string>("direction_topic", "/vision_line_direction_out");

        corner_delay_s_ = nh_private_.param<double>("corner_delay_s", 1.5);
        turning_end_x_error_abs_max_ = nh_private_.param<double>("turning_end_x_error_abs_max", 15.0);
        target_y_ = nh_private_.param<double>("vision_target_y", 400.0);
        turning_target_y_ = nh_private_.param<double>("turning_target_y", 360.0);
        loop_rate_ = nh_private_.param<int>("loop_rate", 120);
        std::string initial_direction = nh_private_.param<std::string>("initial_direction", "stop");

        // 设置初始方向
        if (initial_direction == "straight")
            direction_ = "straight";
        else if (initial_direction == "right")
            direction_ = "right";
        else if (initial_direction == "left")
            direction_ = "left";
        else
            direction_ = "stop";

        // 订阅图像话题
        image_sub_ = it_.subscribe(image_topic, 1, &FindWayROS::imageCallback, this);
        ROS_INFO("Subscribed image topic: %s", image_topic.c_str());

        // 订阅方向话题（全局话题）
        direction_sub_ = nh_.subscribe(direction_topic_, 10, &FindWayROS::directionCallback, this);
        ROS_INFO("Subscribed direction topic: %s", direction_topic_.c_str());

        // 发布视觉线话题（全局话题）
        vision_line_pub_ = nh_.advertise<std_msgs::Float32MultiArray>(vision_line_topic, 10);
        ROS_INFO("Publishing vision line topic: %s", vision_line_topic.c_str());

        // 初始化转弯标志
        ros::param::set(turning_flag_param_, 0);

        // 由 straight_track_side_ 一次性派生出搜索侧与中线模式（运行前算一次，避免每帧 switch）
        switch (straight_track_side_)
        {
        case LEFT_ONLY:
            straight_side_ = LEFT_ONLY;
            straight_mode_ = LEFT_OFFSET;
            break;
        case RIGHT_ONLY:
            straight_side_ = RIGHT_ONLY;
            straight_mode_ = RIGHT_OFFSET;
            break;
        default:
            straight_side_ = BOTH;
            straight_mode_ = MID_AVG;
            break;
        }
    }

    void run()
    {
        ros::Rate rate(loop_rate_);
        ros::Time t0;
        bool t0_set = false;
        double fps = 0.0;
        int frame_count = 0;
        double elapsed = 0.0;
        ros::Time prev_t = ros::Time::now();

        while (ros::ok())
        {
            ros::spinOnce();

            cv::Mat frame;
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                if (latest_frame_.empty())
                {
                    rate.sleep();
                    continue;
                }
                frame = latest_frame_.clone();
            }

            // 计算 FPS
            ros::Time cur_t = ros::Time::now();
            double dt = (cur_t - prev_t).toSec();
            prev_t = cur_t;
            frame_count++;
            elapsed += dt;
            if (frame_count % 30 == 0 && elapsed > 1e-6)
            {
                fps = 30.0 / elapsed;
                elapsed = 0.0;
            }

            // 检查是否需要重置状态机
            {
                std::lock_guard<std::mutex> lock(direction_mutex_);
                if (reset_requested_)
                {
                    state_ = IDLE;
                    t0_set = false;
                    miss_line_ = NO_MISS;
                    processor_.set_mid_line_mode(MID_AVG);
                    reset_requested_ = false;
                    ROS_INFO("State machine reset to IDLE (direction changed)");
                }
            }

            // 原始图像尺寸（用于消息缩放）
            int orig_h = frame.rows;
            int orig_w = frame.cols;

            if (state_ == IDLE)
            {
                std::string dir;
                {
                    std::lock_guard<std::mutex> lock(direction_mutex_);
                    dir = direction_;
                }

                if (dir == "straight")
                {
                    state_ = STRAIGHT_TRACKING;
                    t0 = ros::Time::now();
                    t0_set = true;
                    ros::param::set("/start_vision1", 1);
                    ROS_INFO("State: IDLE -> STRAIGHT_TRACKING");
                }
                else if (dir == "right")
                {
                    state_ = RIGHT_TRACKING;
                    ros::param::set("/start_vision1", 1);
                    ROS_INFO("State: IDLE -> RIGHT_TRACKING");
                }
                else if (dir == "left")
                {
                    state_ = LEFT_TRACKING;
                    ros::param::set("/start_vision1", 1);
                    ROS_INFO("State: IDLE -> LEFT_TRACKING");
                }
            }
            else
            {
                processor_.resize_frame(frame);

                // STRAIGHT_TRACKING 时先对图片做透视变换
                cv::Mat pers_frame;
                if (state_ == STRAIGHT_TRACKING)
                {

                    pers_frame = CameraCapture::perspectiveFrame(frame);
                    if (pers_frame.empty())
                    {
                        std::cerr << "Error: Perspective transform failed, using original frame." << std::endl;
                        pers_frame = frame; // 使用原始帧作为备用
                    }
                }

                else
                {
                    pers_frame = frame;
                }

                processor_.set_frame(pers_frame);

                if (pers_frame.empty())
                {
                    ROS_WARN("Perspective frame is empty, skipping processing.");
                    // rate.sleep();
                    continue;
                }

                cv::Mat binary_img = processor_.preprocess(pers_frame);
                int proc_h = binary_img.rows;
                int proc_w = binary_img.cols;

                // 调试：检查图像尺寸
                // static int debug_count = 0;
                // if (debug_count++ % 30 == 0)
                // {
                //     ROS_INFO("Frame dimensions: orig_h=%d, orig_w=%d, proc_h=%d, proc_w=%d",
                //              orig_h, orig_w, proc_h, proc_w);
                // }

                // 用于消息缩放的比例
                double scale_x = static_cast<double>(orig_w) / proc_w;
                double scale_y = static_cast<double>(orig_h) / proc_h;

                if (state_ == RIGHT_TRACKING || state_ == LEFT_TRACKING)
                {
                    cv::Mat canvas = processor_.return_frame();
                    processor_.get_side_line_task_1(binary_img, canvas, false);
                    processor_.fit_polynomial();

                    auto msg = buildVisionLineMsg(processor_.get_fit_mid_line(),
                                                  proc_h, proc_w,
                                                  orig_h, orig_w,
                                                  config_.left_target_p_index);
                    vision_line_pub_.publish(msg);

                    processor_.draw_line(canvas, fps, state_name(state_));
                    cv::imshow("binary", binary_img);
                    cv::imshow("processed_img", canvas);
                }
                else
                {
                    // STRAIGHT_TRACKING
                    SearchSide side = BOTH;
                    MidLineMode mode = MID_AVG;
                    if (state_ == STRAIGHT_TRACKING)
                    {
                        side = straight_side_;
                        mode = straight_mode_;
                    }
                    processor_.set_mid_line_mode(mode);

                    cv::Mat canvas = processor_.return_frame();
                    processor_.get_side_line_task_2(binary_img, canvas, true, false, side);
                    processor_.calculate_mid_line(binary_img);
                    processor_.fit_polynomial();

                    auto msg = buildVisionLineMsg(processor_.get_fit_mid_line(),
                                                  proc_h, proc_w,
                                                  orig_h, orig_w,
                                                  config_.straight_target_p_index);
                    vision_line_pub_.publish(msg);

                    processor_.draw_line(canvas, fps, state_name(state_));
                    cv::imshow("binary", binary_img);
                    cv::imshow("processed_img", canvas);
                }
            }

            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q')
            {
                ROS_INFO("User quit");
                break;
            }

            rate.sleep();
        }

        ros::param::set(turning_flag_param_, 0);
        ROS_INFO("Reset turning flag param: %s=0", turning_flag_param_.c_str());
        cv::destroyAllWindows();
    }

private:
    ros::NodeHandle nh_;         // 公共节点句柄，用于订阅和发布全局话题
    ros::NodeHandle nh_private_; // 私有节点句柄，用于读取参数
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    ros::Subscriber direction_sub_;
    ros::Publisher vision_line_pub_;

    std::mutex frame_mutex_;
    cv::Mat latest_frame_;

    std::mutex direction_mutex_;
    std::string direction_ = "left";
    bool reset_requested_ = false;

    State state_ = IDLE;
    MissLineState miss_line_ = NO_MISS;

    ImageProcessConfig config_;
    ImageProcess processor_{config_};

    std::string turning_flag_param_;
    std::string direction_topic_;

    double corner_delay_s_ = 1.5;
    double turning_end_x_error_abs_max_ = 15.0;
    double target_y_ = 400.0;
    double turning_target_y_ = 360.0;
    int loop_rate_ = 120;

    // STRAIGHT_TRACKING 走哪一边：LEFT_ONLY / RIGHT_ONLY / BOTH
    SearchSide straight_track_side_ = LEFT_ONLY;
    // 由 straight_track_side_ 派生（构造时算一次）
    SearchSide straight_side_ = BOTH;
    MidLineMode straight_mode_ = MID_AVG;

    void imageCallback(const sensor_msgs::ImageConstPtr &msg)
    {
        try
        {
            cv::Mat frame = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
            std::lock_guard<std::mutex> lock(frame_mutex_);
            latest_frame_ = frame;
        }
        catch (cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

    void directionCallback(const std_msgs::String::ConstPtr &msg)
    {
        std::string dir = msg->data;
        // 去除首尾空白
        size_t start = dir.find_first_not_of(" \t\n\r");
        size_t end = dir.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && end != std::string::npos)
            dir = dir.substr(start, end - start + 1);
        // 转小写
        std::transform(dir.begin(), dir.end(), dir.begin(), ::tolower);

        std::lock_guard<std::mutex> lock(direction_mutex_);
        if (dir != "straight" && dir != "right" && dir != "left" && dir != "stop")
        {
            ROS_WARN("Ignoring invalid direction: '%s'", msg->data.c_str());
            return;
        }

        if (dir == "stop")
        {
            direction_ = "stop";
        }
        else
        {
            direction_ = dir;
        }
        reset_requested_ = true;
        ROS_INFO("Direction set to: %s", dir.c_str());
    }

    std_msgs::Float32MultiArray buildVisionLineMsg(
        const std::vector<cv::Point> &line_points,
        int proc_h, int proc_w,
        int orig_h, int orig_w,
        int target_index)
    {
        std_msgs::Float32MultiArray msg;

        // 调试：检查输入参数的合理性
        // ROS_INFO("buildVisionLineMsg params: proc_h=%d, proc_w=%d, orig_h=%d, orig_w=%d, target_index=%d",
        //          proc_h, proc_w, orig_h, orig_w, target_index);

        if (line_points.empty() ||
            proc_h <= 0 || proc_w <= 0 ||
            orig_h <= 0 || orig_w <= 0 ||
            static_cast<int>(line_points.size()) < std::abs(target_index) + 1)
        {
            msg.data = {0.0, -1.0};
            return msg;
        }

        double scale_x = static_cast<double>(orig_w) / proc_w;
        double scale_y = static_cast<double>(orig_h) / proc_h;

        // 处理负索引（从末尾数）
        int idx = target_index;
        if (idx < 0)
        {
            idx = static_cast<int>(line_points.size()) + idx;
        }

        if (idx < 0 || idx >= static_cast<int>(line_points.size()))
        {
            msg.data = {0.0, -1.0};
            return msg;
        }

        cv::Point pt = line_points[idx];
        double x_raw = pt.x * scale_x;
        double y_raw = pt.y * scale_y;
        double x_error = x_raw - (orig_w / 2.0);
        // ROS_INFO(" ptx: %.2f, pty: %.2f, x_raw: %.2f, x_error: %.2f, y_raw: %.2f, idx: %d", static_cast<double>(pt.x), static_cast<double>(pt.y), x_raw, x_error, y_raw, target_index);

        msg.data = {static_cast<float>(x_error), static_cast<float>(y_raw)};
        return msg;
    }

    // static std::string stateToStr(State s)
    // {
    //     switch (s)
    //     {
    //     case IDLE:
    //         return "IDLE";
    //     case STRAIGHT_TRACKING:
    //         return "STRAIGHT_TRACKING";
    //     case RIGHT_TRACKING:
    //         return "RIGHT_TRACKING";
    //     case LEFT_TRACKING:
    //         return "LEFT_TRACKING";
    //     case CORNER:
    //         return "CORNER";
    //     case CROSS:
    //         return "CROSS";
    //     case TURNING:
    //         return "TURNING";
    //     case TRACKING2:
    //         return "TRACKING2";
    //     default:
    //         return "UNKNOWN";
    //     }
    // }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "find_way_ros");
    FindWayROS node;
    node.run();
    return 0;
}
