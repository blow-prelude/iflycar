#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/String.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <mutex>

#include "image_process.h"
#include "ground_perspective.h"

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
        turning_end_confirm_frames_ = std::max(1, nh_private_.param<int>("turning_end_confirm_frames", 3));
        stop_line_end_y_thresh_ = nh_private_.param<double>("stop_line_end_y_thresh", 0.55);
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

        // 由 straight_track_side_ 一次性派生转弯前、转弯后的搜索侧与中线模式
        // （运行前算一次，避免每帧 switch）
        straight_side_ = straight_track_side_;
        straight_mode_ = midLineModeForSide(straight_side_);
        post_turn_side_ = oppositeSide(straight_side_);
        post_turn_mode_ = midLineModeForSide(post_turn_side_);
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
                    resetTurningState();
                    // 清空消息缓存，避免把上一任务的控制误差带入新任务
                    has_last_valid_vision_line_msg_ = false;
                    last_valid_vision_line_msg_.data.clear();
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
                    // 初始即视为转弯进行中：进入直行巡线即置位转弯标志，
                    // 由水平白线判定结束后再清零。
                    ros::param::set(turning_flag_param_, 1);
                    ROS_INFO("State: IDLE -> STRAIGHT_TRACKING (%s=1)", turning_flag_param_.c_str());
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
                // STRAIGHT_TRACKING 时先对图片做透视变换
                cv::Mat pers_frame;
                if (state_ == STRAIGHT_TRACKING)
                {
                    // 透视变换前保留原有的 320x240 输入缩放；普通帧的缩放
                    // 由 preprocess 统一完成。
                    cv::Mat perspective_input = frame;
                    processor_.resize_frame(perspective_input);
                    cv::imshow("resized_input", perspective_input);
                    try
                    {
                        // 固定透视矩阵要求输入为 320x240；resize_frame 已保证该尺寸。
                        pers_frame = vision_line::warpFixedGroundPerspective(perspective_input);
                    }
                    catch (const std::exception &e)
                    {
                        ROS_WARN("warpFixedGroundPerspective failed (%s); using resized frame.", e.what());
                        pers_frame = perspective_input; // 使用缩放后的帧作为备用
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

                // preprocess 负责按配置缩放、有效区域背景估计、二值化和闭运算。
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
                    processor_.get_side_line_task_1(binary_img, canvas, true);
                    processor_.fit_polynomial2();

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
                    cv::Mat canvas = processor_.return_frame();

                    if (turn_completed_)
                    {
                        // 转弯结束后常驻相反单边，并采用与该侧匹配的偏移中线模式。
                        processor_.set_mid_line_mode(post_turn_mode_);
                        processor_.get_side_line_task_2(binary_img, canvas, true, false,
                                                        post_turn_side_, straight_side_);
                    }
                    else
                    {
                        // 转弯进行中：初始侧巡线，不再检测拐点（find_corner=false）。
                        processor_.set_mid_line_mode(straight_mode_);
                        processor_.get_side_line_task_2(binary_img, canvas, true, false, straight_side_);

                        // 用水平白线判定转弯是否结束；刚结束则同帧切到相反侧巡线，
                        // 避免中线切换延迟到下一帧。
                        if (checkStopLineTurnEnd(binary_img, canvas))
                        {
                            processor_.set_mid_line_mode(post_turn_mode_);
                            processor_.get_side_line_task_2(binary_img, canvas, true, false,
                                                            post_turn_side_, straight_side_);
                        }
                    }

                    // 本节点只在顶部按斜率向上补线，不向下补齐边线。
                    processor_.calculate_mid_line(binary_img, false);
                    processor_.fit_polynomial();

                    auto msg = buildVisionLineMsg(processor_.get_fit_mid_line(),
                                                  proc_h, proc_w,
                                                  orig_h, orig_w,
                                                  config_.straight_target_p_index);
                    vision_line_pub_.publish(msg);

                    processor_.draw_line(canvas, fps, state_name(state_));
                    // cv::imshow("perspective", pers_frame);
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
    int turning_end_confirm_frames_ = 3;
    double stop_line_end_y_thresh_ = 0.55;
    int loop_rate_ = 120;

    // 初始即视为转弯进行中，只判定结束；turn_completed_ 置位后常驻相反侧巡线
    bool turn_completed_ = false;
    int stop_line_end_count_ = 0;

    // STRAIGHT_TRACKING 走哪一边：LEFT_ONLY / RIGHT_ONLY / BOTH
    SearchSide straight_track_side_ = LEFT_ONLY;
    // 由 straight_track_side_ 派生（构造时算一次）
    SearchSide straight_side_ = BOTH;
    MidLineMode straight_mode_ = MID_AVG;
    // 转弯结束后改用的相反巡线侧与中线模式（由 straight_side_ 派生）
    SearchSide post_turn_side_ = BOTH;
    MidLineMode post_turn_mode_ = MID_AVG;

    // 缓存最后一条有效的 /vision_line 消息，丢点时复用，保证控制连续
    std_msgs::Float32MultiArray last_valid_vision_line_msg_;
    bool has_last_valid_vision_line_msg_ = false;

    // SearchSide -> MidLineMode 映射，避免在构造函数和帧循环中重复 switch
    static MidLineMode midLineModeForSide(SearchSide side)
    {
        if (side == LEFT_ONLY)
            return LEFT_OFFSET;
        if (side == RIGHT_ONLY)
            return RIGHT_OFFSET;
        return MID_AVG;
    }

    // 左右单边反转；BOTH 无相反侧，保持双边平均
    static SearchSide oppositeSide(SearchSide side)
    {
        if (side == LEFT_ONLY)
            return RIGHT_ONLY;
        if (side == RIGHT_ONLY)
            return LEFT_ONLY;
        return BOTH;
    }

    // 用水平白线判定转弯是否结束：中点 y 超过阈值连续 turning_end_confirm_frames_ 帧即结束，
    // 结束时清零对外转弯标志。初始即视为转弯进行中，无“开始”事件。
    // 每帧在 canvas 上绘制停止线中点与阈值参考线，便于标定 stop_line_end_y_thresh_。
    // 返回 true 表示本次调用刚判定转弯结束（调用方需同帧切到相反侧巡线）。
    bool checkStopLineTurnEnd(cv::Mat &binary_img, cv::Mat &canvas)
    {
        if (turn_completed_)
            return false;

        std::vector<int> stop_line = processor_.get_stop_line(binary_img, canvas, true);

        bool stop_line_low = false;
        if (!stop_line.empty() && binary_img.rows > 0)
        {
            const float y_norm = stop_line[1] / static_cast<float>(binary_img.rows);
            stop_line_low = y_norm > stop_line_end_y_thresh_;

            // 可视化：阈值参考线（绿）+ 中点高亮（黄环），便于观察中点何时越过阈值
            if (canvas.rows > 0)
            {
                const int thresh_y = static_cast<int>(stop_line_end_y_thresh_ * canvas.rows);
                cv::line(canvas, cv::Point(0, thresh_y), cv::Point(canvas.cols, thresh_y),
                         cv::Scalar(0, 255, 0), 1);
                const cv::Point center(stop_line[0], stop_line[1]);
                cv::circle(canvas, center, 8, cv::Scalar(0, 255, 255), 2);
            }
        }

        if (stop_line_low)
        {
            if (stop_line_end_count_ < turning_end_confirm_frames_)
                ++stop_line_end_count_;
        }
        else
        {
            stop_line_end_count_ = 0;
        }

        if (stop_line_end_count_ >= turning_end_confirm_frames_)
        {
            turn_completed_ = true;
            stop_line_end_count_ = 0;
            ros::param::set(turning_flag_param_, 0);
            ROS_INFO("Stop line midpoint y_norm > %.2f for %d consecutive frames; turning finished (%s=0)",
                     stop_line_end_y_thresh_, turning_end_confirm_frames_, turning_flag_param_.c_str());
            return true;
        }
        return false;
    }

    void resetTurningState()
    {
        turn_completed_ = false;
        stop_line_end_count_ = 0;
        ros::param::set(turning_flag_param_, 0);
    }

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

    // 当前帧无法取得目标点时的统一回退：有缓存则复用上一条有效消息，
    // 否则发送 [0.0, -1.0] 表示无数据。
    std_msgs::Float32MultiArray lastValidOrInvalidMsg() const
    {
        if (has_last_valid_vision_line_msg_)
            return last_valid_vision_line_msg_;

        std_msgs::Float32MultiArray msg;
        msg.data = {0.0f, -1.0f};
        return msg;
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
            return lastValidOrInvalidMsg();
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
            return lastValidOrInvalidMsg();
        }

        cv::Point pt = line_points[idx];
        double x_raw = pt.x * scale_x;
        double y_raw = pt.y * scale_y;
        double x_error = x_raw - (orig_w / 2.0);
        // ROS_INFO(" ptx: %.2f, pty: %.2f, x_raw: %.2f, x_error: %.2f, y_raw: %.2f, idx: %d", static_cast<double>(pt.x), static_cast<double>(pt.y), x_raw, x_error, y_raw, target_index);

        msg.data = {static_cast<float>(x_error), static_cast<float>(y_raw)};
        // 成功取得目标点：写入缓存后再返回，丢点时复用本条消息
        last_valid_vision_line_msg_ = msg;
        has_last_valid_vision_line_msg_ = true;
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
