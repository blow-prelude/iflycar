#include <ros/ros.h>
#include "geometry_msgs/Twist.h"
#include "nav_msgs/Odometry.h"
#include "sensor_msgs/LaserScan.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/String.h"
#include "std_srvs/SetBool.h"
#include <algorithm>
#include <limits>
#include <vector>
#include <cmath>
#include <mutex>
#include <stdexcept>

class VisionErrorController
{
private:
    // 配置参数
    const double MAX_ANGULAR_VEL = 1.0;    // 最大角速度 (rad/s)
    const double MAX_LINEAR_VEL = 0.5;     // 最大线速度 (m/s)
    const double MIN_LINEAR_VEL = 0.05;    // 最小线速度 (m/s)
    const int PIXEL_ERROR_THRESHOLD = 10;  // x 误差收敛阈值
    const int STABLE_COUNT_THRESHOLD = 5;  // 误差稳定计数阈值
    const int Y_LOWER_BOUND = 225;         // y 值有效范围下限
    const int Y_UPPER_BOUND = 460;         // y 值有效范围上限
    const double X_REFERENCE = 0.0;        // x 方向参考值
    const double Y_ERROR_TOLERANCE = 0.05; // y方向位置误差容忍值 (m)
    const double LOOP_RATE = 50.0;         // 主循环频率 (Hz)

    // 巡线避障机动距离。速度通过私有参数配置，按速度×时长给出名义位移。
    const double AVOIDANCE_LATERAL_DISTANCE = 0.5;
    const double AVOIDANCE_FORWARD_DISTANCE = 0.65;
    const double FORCE_YAW_TURN_TARGET = -1.57;   // left/right 避障后的绝对目标航向 (rad)
    const double FORCE_YAW_STRAIGHT_TARGET = 0.0; // straight 避障后的绝对目标航向 (rad)
    const double FORCE_YAW_KP = 2.0;              // 参考 GotoA 的 Kp_yaw
    const double FORCE_YAW_TOLERANCE = 0.05;      // 航向收敛阈值 (rad)

    // 方向机动状态机
    enum class ManeuverState
    {
        IDLE,
        FORWARD,
        ROTATE,
        DONE
    };
    ManeuverState maneuver_state_ = ManeuverState::IDLE;
    ros::Time maneuver_start_time_;
    double maneuver_direction_; // 1=left(逆时针), -1=right(顺时针)
    bool needs_rotate_;         // straight 不需要旋转
    double forward_duration_;   // 0.35 / 0.3 ≈ 1.167s
    double rotate_duration_;    // (75° * π/180) / angular_vel
    ros::Subscriber direction_sub_;
    ros::Publisher direction_pub_;
    std::string last_direction_ = "straight"; // 存储收到的方向，供转发
    bool direction_received_ = false;         // 仅收到有效 direction 后才允许触发避障

    // 巡线期间的避障状态机（进程内最多触发一次）：
    // 停车 -> 横移避开 -> 前进 -> 横移回线 -> 等待前方清空 -> 强制航向。
    enum class AvoidanceState
    {
        IDLE,
        STOP,
        LATERAL_OUT,
        MOVE_FORWARD,
        LATERAL_BACK,
        WAIT_CLEAR,
        FORCE_YAW
    };
    AvoidanceState avoidance_state_ = AvoidanceState::IDLE;
    bool avoidance_used_ = false; // 本节点进程生命周期内只允许触发一次雷达避障
    ros::Time avoidance_start_time_;
    double avoidance_lateral_direction_ = 1.0; // +1=左移，-1=右移（底盘+Y为左）
    double avoidance_lateral_duration_ = 0.0;
    double avoidance_forward_duration_ = 0.0;
    double force_yaw_target_ = FORCE_YAW_STRAIGHT_TARGET;

    // 雷达前方障碍检测缓存。
    ros::Subscriber scan_sub_;
    ros::Subscriber odom_sub_;
    ros::Publisher avoidance_active_pub_;
    bool front_obstacle_detected_ = false;
    bool front_scan_valid_ = false;
    double front_obstacle_distance_ = std::numeric_limits<double>::infinity();
    ros::Time last_scan_time_;
    double obstacle_distance_threshold_ = 0.35;
    double obstacle_front_half_angle_ = 20.0 * M_PI / 180.0;
    double obstacle_scan_timeout_ = 0.5;
    double avoidance_lateral_speed_ = 0.2;
    double avoidance_forward_speed_ = 0.3;
    double avoidance_stop_duration_ = 0.15;
    double current_yaw_ = 0.0;
    bool yaw_valid_ = false;
    ros::Time last_odom_time_;

    // PID 控制器结构体
    struct PIDController
    {
        double kp, ki, kd;
        double err, err_last, err_sum;
        double output_limit;
    };

    // 视觉数据结构体
    struct VisionData
    {
        float x;       // x 误差
        float y;       // y 像素坐标
        bool new_data; // 新数据标志
        bool valid;    // 数据有效性标志
    };

    // 类成员变量
    ros::NodeHandle nh_; // ROS节点句柄
    ros::Publisher cmd_vel_pub_;
    ros::NodeHandle nh_private_{"~"};
    ros::ServiceServer enable_service_;
    bool enabled_ = false;
    double disabled_rate_ = 10.0;
    ros::Subscriber vision_sub_;
    PIDController angular_pid_; // 角速度PID控制器（按方向切换参数）
    PIDController linear_pid_;  // 线速度PID控制器
    // straight 用 angular_pid_ 的现参数；left/right 用另一套。
    std::string angular_pid_dir_; // 当前 angular_pid_ 装载的方向
    VisionData vision_data_;
    int start_vision_;
    int start_vision_line2_;
    int prev_start_vision_line2_;
    bool turning_mode_;
    bool is_stopped_;
    double current_error_;
    int stable_count_;
    double target_y_;  // 目标y坐标（来自switch.cpp）
    double current_y_; // 当前y坐标（来自参数服务器）
    double turning_angular_vel_;
    ros::Time turn_start_time_;
    std::mutex data_mutex_;
    const double FIXED_TURN_DURATION = 1.8; // 固定旋转时长 (s)

    // 初始化PID参数
    void initPID()
    {
        // 角速度PID参数（straight 初始装载）
        angular_pid_ = {0.007, 0.0, 0.007, 0, 0, 0, MAX_ANGULAR_VEL};
        angular_pid_dir_ = "straight";
        // 线速度PID参数
        linear_pid_ = {0.5, 0.01, 0.05, 0, 0, 0, MAX_LINEAR_VEL};
    }

    // 按方向切换角速度PID参数；方向变化时清空积分与微分记忆。
    // straight 用现参数；left/right 用另一套（待调）。
    void applyAngularPIDForDir(const std::string &dir)
    {
        if (dir == angular_pid_dir_)
            return;
        if (dir == "straight")
            angular_pid_ = {0.007, 0.0, 0.007, 0, 0, 0, MAX_ANGULAR_VEL};
        else // left / right
            angular_pid_ = {0.007, 0.0, 0.007, 0, 0, 0, MAX_ANGULAR_VEL};
        angular_pid_dir_ = dir;
    }

    // PID计算函数（通用）
    double calculatePID(double err, PIDController &pid)
    {
        pid.err = err;

        // 误差较小时累加积分项
        if (fabs(pid.err) < 20.0)
        {
            pid.err_sum += pid.err;
            pid.err_sum = std::max(std::min(pid.err_sum, 50.0), -50.0); // 积分限幅
        }
        else
        {
            pid.err_sum = 0; // 误差过大时清空积分
        }

        double diff = pid.err - pid.err_last; // 微分项
        double output = pid.kp * pid.err + pid.ki * pid.err_sum + pid.kd * diff;
        output = std::max(std::min(output, pid.output_limit), -pid.output_limit); // 输出限幅
        pid.err_last = pid.err;

        return output;
    }

    // 计算x方向线速度PID
    double calculateLinearPID(double err)
    {
        linear_pid_.err = err;

        // 积分限幅
        if (fabs(linear_pid_.err) < 0.1)
        { // 误差较小时累加积分
            linear_pid_.err_sum += linear_pid_.err;
            linear_pid_.err_sum = std::max(std::min(linear_pid_.err_sum, 0.5), -0.5);
        }
        else
        {
            linear_pid_.err_sum = 0;
        }

        // 微分项
        double diff = linear_pid_.err - linear_pid_.err_last;
        double output = linear_pid_.kp * linear_pid_.err +
                        linear_pid_.ki * linear_pid_.err_sum +
                        linear_pid_.kd * diff;

        // 输出限幅（确保最小速度）
        output = std::max(std::min(output, linear_pid_.output_limit), MIN_LINEAR_VEL);
        linear_pid_.err_last = linear_pid_.err;

        return output;
    }

    // 清除当前避障机动。禁用巡线或收到 stop 时调用，避免下次启动时接着执行旧状态。
    void resetAvoidance()
    {
        const bool was_active = avoidance_state_ != AvoidanceState::IDLE;
        // 这里只重置当前阶段；avoidance_used_ 故意保留，保证进程内只执行一次。
        avoidance_state_ = AvoidanceState::IDLE;
        avoidance_start_time_ = ros::Time(0);
        avoidance_lateral_direction_ = 1.0;
        force_yaw_target_ = FORCE_YAW_STRAIGHT_TARGET;
        if (was_active)
        {
            std_msgs::Bool msg;
            msg.data = false;
            avoidance_active_pub_.publish(msg);
        }
    }

    // 将角度归一化到 [-pi, pi]，用于绝对航向误差计算。
    double normalizeAngle(double angle) const
    {
        while (angle > M_PI)
            angle -= 2.0 * M_PI;
        while (angle < -M_PI)
            angle += 2.0 * M_PI;
        return angle;
    }

    // /odom 回调：提取当前车体 yaw，和 GotoA 中的 yaw 反馈作用相同。
    void odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
    {
        const geometry_msgs::Quaternion &q = msg->pose.pose.orientation;
        const double sin_yaw = 2.0 * (q.w * q.z + q.x * q.y);
        const double cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        const double yaw = std::atan2(sin_yaw, cos_yaw);
        if (!std::isfinite(yaw))
            return;

        std::lock_guard<std::mutex> lock(data_mutex_);
        current_yaw_ = yaw;
        yaw_valid_ = true;
        last_odom_time_ = ros::Time::now();
    }

    // 优先使用新鲜 /odom；若车上已有 TransformListener2，也允许用 CarYaw 参数兜底。
    bool getCurrentYaw(double &yaw)
    {
        bool odom_valid = false;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            const double odom_age = (ros::Time::now() - last_odom_time_).toSec();
            odom_valid = yaw_valid_ && !last_odom_time_.isZero() &&
                         odom_age >= 0.0 && odom_age <= obstacle_scan_timeout_;
            if (odom_valid)
                yaw = current_yaw_;
        }
        if (odom_valid)
            return true;

        double parameter_yaw = 0.0;
        if (nh_.getParam("CarYaw", parameter_yaw) && std::isfinite(parameter_yaw))
        {
            yaw = parameter_yaw;
            return true;
        }
        return false;
    }

    // 完成避障后的公共收尾：清 PID/视觉缓存并恢复巡线。
    void finishAvoidance()
    {
        avoidance_state_ = AvoidanceState::IDLE;
        current_error_ = 0.0;
        angular_pid_.err_last = 0.0;
        angular_pid_.err_sum = 0.0;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            vision_data_ = {0.0f, 0.0f, false, false};
        }
        std_msgs::Bool active_msg;
        active_msg.data = false;
        avoidance_active_pub_.publish(active_msg);
        ROS_INFO("Avoidance complete: resume line following");
    }

    // 读取雷达缓存，并检查数据是否仍在有效时间内。
    void getFrontObstacleState(bool &detected, bool &valid,
                               double &distance, bool &fresh)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        detected = front_obstacle_detected_;
        valid = front_scan_valid_;
        distance = front_obstacle_distance_;
        const double scan_age = (ros::Time::now() - last_scan_time_).toSec();
        fresh = !last_scan_time_.isZero() && scan_age >= 0.0 &&
                scan_age <= obstacle_scan_timeout_;
    }

    // /scan 回调：只在车体正前方角度窗口内取最近的有效量程。
    void scanCallback(const sensor_msgs::LaserScan::ConstPtr &msg)
    {
        double min_range = std::numeric_limits<double>::infinity();
        bool has_valid_range = false;

        if (msg->angle_increment > 0.0)
        {
            const double min_range_limit = std::max(0.0, static_cast<double>(msg->range_min));
            const double max_range_limit = static_cast<double>(msg->range_max);
            const bool has_range_max = std::isfinite(max_range_limit) &&
                                       max_range_limit > min_range_limit;
            for (std::size_t i = 0; i < msg->ranges.size(); ++i)
            {
                double angle = msg->angle_min +
                               static_cast<double>(i) * msg->angle_increment;
                while (angle > M_PI)
                    angle -= 2.0 * M_PI;
                while (angle < -M_PI)
                    angle += 2.0 * M_PI;

                if (std::fabs(angle) > obstacle_front_half_angle_)
                    continue;

                const double range = msg->ranges[i];
                if (!std::isfinite(range) || range < min_range_limit ||
                    (has_range_max && range > max_range_limit))
                    continue;

                has_valid_range = true;
                min_range = std::min(min_range, range);
            }
        }

        std::lock_guard<std::mutex> lock(data_mutex_);
        front_scan_valid_ = has_valid_range;
        front_obstacle_distance_ = min_range;
        front_obstacle_detected_ = has_valid_range &&
                                   min_range <= obstacle_distance_threshold_;
        last_scan_time_ = ros::Time::now();
    }

    // 根据 direction 选择横移方向并启动避障。straight/right 左移，left 右移。
    void startAvoidance(double obstacle_distance)
    {
        if (last_direction_ == "left")
            avoidance_lateral_direction_ = -1.0;
        else if (last_direction_ == "straight" || last_direction_ == "right")
            avoidance_lateral_direction_ = 1.0;
        else
            return;

        // 在首次进入避障状态时消耗唯一一次机会；resetAvoidance() 不会清除此标志。
        avoidance_used_ = true;
        force_yaw_target_ = last_direction_ == "straight"
                                ? FORCE_YAW_STRAIGHT_TARGET
                                : FORCE_YAW_TURN_TARGET;
        avoidance_state_ = AvoidanceState::STOP;
        avoidance_start_time_ = ros::Time::now();
        std_msgs::Bool active_msg;
        active_msg.data = true;
        avoidance_active_pub_.publish(active_msg);
        ROS_WARN("Front obstacle detected (%.3fm). Avoidance: stop, %s %.2fm, "
                 "forward %.2fm, return %.2fm",
                 obstacle_distance,
                 avoidance_lateral_direction_ > 0.0 ? "left" : "right",
                 AVOIDANCE_LATERAL_DISTANCE,
                 AVOIDANCE_FORWARD_DISTANCE,
                 AVOIDANCE_LATERAL_DISTANCE);
    }

    // 处理避障状态；返回 true 表示本周期不应再执行视觉巡线控制。
    bool processAvoidance(geometry_msgs::Twist &cmd)
    {
        bool obstacle_detected = false;
        bool scan_valid = false;
        bool scan_fresh = false;
        double obstacle_distance = std::numeric_limits<double>::infinity();
        getFrontObstacleState(obstacle_detected, scan_valid, obstacle_distance, scan_fresh);

        if (avoidance_state_ == AvoidanceState::IDLE)
        {
            const bool direction_valid = direction_received_ &&
                                         (last_direction_ == "left" ||
                                          last_direction_ == "right" ||
                                          last_direction_ == "straight");
            if (!avoidance_used_ && scan_valid && scan_fresh &&
                obstacle_detected && direction_valid)
            {
                startAvoidance(obstacle_distance);
                cmd = geometry_msgs::Twist();
                return true;
            }
            if (scan_valid && scan_fresh && obstacle_detected && !direction_valid)
            {
                ROS_WARN_THROTTLE(
                    1.0,
                    "Front obstacle detected, but no active direction is available (last='%s')",
                    last_direction_.c_str());
                cmd = geometry_msgs::Twist();
                return true;
            }
            if (avoidance_used_ && scan_valid && scan_fresh && obstacle_detected)
            {
                ROS_DEBUG_THROTTLE(1.0,
                                   "Front obstacle ignored: one-shot avoidance already used");
            }
            return false;
        }

        // 所有避障阶段都覆盖视觉输出，避免角速度/PID 与横移叠加。
        cmd = geometry_msgs::Twist();
        const double elapsed =
            (ros::Time::now() - avoidance_start_time_).toSec();

        switch (avoidance_state_)
        {
        case AvoidanceState::STOP:
            if (elapsed >= avoidance_stop_duration_)
            {
                avoidance_state_ = AvoidanceState::LATERAL_OUT;
                avoidance_start_time_ = ros::Time::now();
                ROS_INFO("Avoidance: STOP -> LATERAL_OUT");
            }
            break;

        case AvoidanceState::LATERAL_OUT:
            cmd.linear.y = avoidance_lateral_direction_ * avoidance_lateral_speed_;
            if (elapsed >= avoidance_lateral_duration_)
            {
                avoidance_state_ = AvoidanceState::MOVE_FORWARD;
                avoidance_start_time_ = ros::Time::now();
                ROS_INFO("Avoidance: LATERAL_OUT -> MOVE_FORWARD");
            }
            break;

        case AvoidanceState::MOVE_FORWARD:
            cmd.linear.x = avoidance_forward_speed_;
            if (elapsed >= avoidance_forward_duration_)
            {
                avoidance_state_ = AvoidanceState::LATERAL_BACK;
                avoidance_start_time_ = ros::Time::now();
                ROS_INFO("Avoidance: MOVE_FORWARD -> LATERAL_BACK");
            }
            break;

        case AvoidanceState::LATERAL_BACK:
            cmd.linear.y = -avoidance_lateral_direction_ * avoidance_lateral_speed_;
            if (elapsed >= avoidance_lateral_duration_)
            {
                avoidance_state_ = AvoidanceState::WAIT_CLEAR;
                avoidance_start_time_ = ros::Time::now();
                ROS_INFO("Avoidance: LATERAL_BACK -> WAIT_CLEAR");
            }
            break;

        case AvoidanceState::WAIT_CLEAR:
            if (scan_valid && scan_fresh && !obstacle_detected)
            {
                avoidance_state_ = AvoidanceState::FORCE_YAW;
                avoidance_start_time_ = ros::Time::now();
                ROS_INFO("Avoidance: front scan is clear, force yaw to %.2f rad",
                         force_yaw_target_);
            }
            break;

        case AvoidanceState::FORCE_YAW:
        {
            double current_yaw = 0.0;
            if (!getCurrentYaw(current_yaw))
            {
                ROS_WARN_THROTTLE(1.0,
                                  "Cannot force yaw to %.2f: no fresh /odom or CarYaw",
                                  force_yaw_target_);
                break;
            }

            const double yaw_error =
                normalizeAngle(force_yaw_target_ - current_yaw);
            if (std::fabs(yaw_error) <= FORCE_YAW_TOLERANCE)
            {
                finishAvoidance();
                break;
            }

            const double yaw_speed_limit = std::min(
                MAX_ANGULAR_VEL,
                std::max(0.05, std::fabs(turning_angular_vel_)));
            cmd.angular.z = std::max(
                -yaw_speed_limit,
                std::min(yaw_speed_limit, FORCE_YAW_KP * yaw_error));
            ROS_INFO_THROTTLE(0.5,
                              "Forcing yaw: current=%.3f target=%.3f error=%.3f cmd=%.3f",
                              current_yaw, force_yaw_target_, yaw_error, cmd.angular.z);
            break;
        }

        case AvoidanceState::IDLE:
            // 已在函数开头处理。
            break;
        }

        return true;
    }

    // 方向指令回调函数
    void directionCallback(const std_msgs::String::ConstPtr &msg)
    {
        if (!enabled_)
            return;
        std::string dir = msg->data;

        // stop 优先级最高：任何机动状态(IDLE/FORWARD/ROTATE/DONE)都立即停车
        if (dir == "stop")
        {
            maneuver_state_ = ManeuverState::DONE;
            resetAvoidance();
            last_direction_ = "stop";
            direction_received_ = false;
            std_msgs::String dir_msg;
            dir_msg.data = last_direction_;
            direction_pub_.publish(dir_msg);
            ROS_INFO("Maneuver: -> DONE (stop)");
            return;
        }

        // 避障机动期间只接受 stop，避免新的方向指令打断横移/前进序列。
        if (avoidance_state_ != AvoidanceState::IDLE)
            return;

        // 卡在 DONE(stop 后未解锁)时，新方向直接复位，不再死等 /start_vision1
        if (maneuver_state_ == ManeuverState::DONE)
        {
            maneuver_state_ = ManeuverState::IDLE;
            resetAvoidance();
            turning_mode_ = false;
            stable_count_ = 0;
            current_error_ = 0.0;
            ROS_INFO("Maneuver: DONE -> IDLE (new direction %s after stop)", dir.c_str());
        }

        if (maneuver_state_ != ManeuverState::IDLE)
            return; // FORWARD/ROTATE 途中忽略非 stop 指令

        if (dir == "left")
        {
            maneuver_direction_ = 1.0;
            needs_rotate_ = true;
            last_direction_ = "left";
        }
        else if (dir == "right")
        {
            maneuver_direction_ = -1.0;
            needs_rotate_ = true;
            last_direction_ = "right";
        }
        else if (dir == "straight")
        {
            maneuver_direction_ = 0.0;
            needs_rotate_ = false;
            last_direction_ = "straight";
        }
        else
        {
            return;
        }
        direction_received_ = true;
        maneuver_state_ = ManeuverState::FORWARD;
        maneuver_start_time_ = ros::Time::now();
        ROS_INFO("Maneuver: IDLE -> FORWARD (direction=%s)", dir.c_str());
    }

    // 视觉数据回调函数
    void visionCallback(const std_msgs::Float32MultiArray::ConstPtr &msg)
    {
        if (!enabled_)
            return;
        std::lock_guard<std::mutex> lock(data_mutex_);

        // 检查数据格式
        if (msg->data.size() != 2)
        {
            vision_data_.valid = false;
            vision_data_.new_data = true;
            static ros::Time last_warn = ros::Time::now();
            if (ros::Time::now() - last_warn > ros::Duration(1.0))
            {
                ROS_WARN("Invalid data format (should be 2 elements), actual: %zu", msg->data.size());
                last_warn = ros::Time::now();
            }
            return;
        }

        // 存储新数据（覆盖旧数据）
        vision_data_.x = msg->data[0];
        vision_data_.y = msg->data[1];
        vision_data_.valid = true;
        vision_data_.new_data = true;
    }

public:
    // 构造函数
    VisionErrorController() : start_vision_(0), start_vision_line2_(0), prev_start_vision_line2_(0),
                              turning_mode_(false), is_stopped_(false),
                              current_error_(0.0), stable_count_(0)
    {
        // 初始化视觉数据
        vision_data_ = {0.0f, 0.0f, false, false};

        // 创建发布者和订阅者
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
        avoidance_active_pub_ =
            nh_.advertise<std_msgs::Bool>("/vision_line_avoidance_active", 1, true);
        vision_sub_ = nh_.subscribe("/vision_line", 10, &VisionErrorController::visionCallback, this);
        scan_sub_ = nh_.subscribe("/scan", 10, &VisionErrorController::scanCallback, this);
        odom_sub_ = nh_.subscribe("/odom", 10, &VisionErrorController::odomCallback, this);
        direction_sub_ = nh_.subscribe("/vision_line_direction", 10, &VisionErrorController::directionCallback, this);
        direction_pub_ = nh_.advertise<std_msgs::String>("/vision_line_direction_out", 10);
        std_msgs::Bool initial_avoidance_state;
        initial_avoidance_state.data = false;
        avoidance_active_pub_.publish(initial_avoidance_state);
        nh_private_.param("disabled_rate", disabled_rate_, 10.0);
        if (disabled_rate_ <= 0.0)
            throw std::invalid_argument("~disabled_rate must be > 0");

        // 避障参数：距离和角度可按实际雷达安装/底盘速度调整。
        nh_private_.param("obstacle_distance_threshold",
                          obstacle_distance_threshold_, 0.35);
        nh_private_.param("obstacle_front_half_angle", // rad
                          obstacle_front_half_angle_, 20.0 * M_PI / 180.0);
        nh_private_.param("obstacle_scan_timeout", obstacle_scan_timeout_, 0.5);
        nh_private_.param("avoidance_lateral_speed", avoidance_lateral_speed_, 0.2);
        nh_private_.param("avoidance_forward_speed", avoidance_forward_speed_, 0.3);
        nh_private_.param("avoidance_stop_duration", avoidance_stop_duration_, 0.15);
        if (!std::isfinite(obstacle_distance_threshold_) ||
            obstacle_distance_threshold_ <= 0.0 ||
            !std::isfinite(obstacle_front_half_angle_) ||
            obstacle_front_half_angle_ <= 0.0 ||
            obstacle_front_half_angle_ > M_PI ||
            !std::isfinite(obstacle_scan_timeout_) || obstacle_scan_timeout_ <= 0.0 ||
            !std::isfinite(avoidance_lateral_speed_) || avoidance_lateral_speed_ <= 0.0 ||
            !std::isfinite(avoidance_forward_speed_) || avoidance_forward_speed_ <= 0.0 ||
            !std::isfinite(avoidance_stop_duration_) || avoidance_stop_duration_ < 0.0)
        {
            throw std::invalid_argument("invalid avoidance/scan parameter");
        }

        // 不允许避障速度超过该节点声明的底盘上限。
        avoidance_lateral_speed_ =
            std::min(avoidance_lateral_speed_, MAX_LINEAR_VEL);
        avoidance_forward_speed_ =
            std::min(avoidance_forward_speed_, MAX_LINEAR_VEL);
        avoidance_lateral_duration_ =
            AVOIDANCE_LATERAL_DISTANCE / avoidance_lateral_speed_;
        avoidance_forward_duration_ =
            AVOIDANCE_FORWARD_DISTANCE / avoidance_forward_speed_;

        enable_service_ = nh_private_.advertiseService(
            "set_enabled", &VisionErrorController::setEnabledCallback, this);

        // 读取参数服务器配置
        turning_angular_vel_ = nh_.param("/turning_angular_vel", 0.5);
        forward_duration_ = 0.33 / 0.3;
        rotate_duration_ = (75.0 * M_PI / 180.0) / turning_angular_vel_;
        initPID();

        // 打印初始化信息
        ROS_INFO("Vision Line Controller started (x-vel + angular control)");
        ROS_INFO("Loop rate: %.1fHz | Valid Y range: [%d, %d]",
                 LOOP_RATE, Y_LOWER_BOUND, Y_UPPER_BOUND);
        ROS_INFO("Max linear vel: %.2fm/s | Max angular vel: %.2frad/s",
                 MAX_LINEAR_VEL, MAX_ANGULAR_VEL);
        ROS_INFO("Turning cmd angular: %.2frad/s", turning_angular_vel_);
        ROS_INFO("Avoidance: threshold=%.2fm, front half-angle=%.1fdeg, "
                 "lateral=%.2fm@%.2fm/s, forward=%.2fm@%.2fm/s",
                 obstacle_distance_threshold_,
                 obstacle_front_half_angle_ * 180.0 / M_PI,
                 AVOIDANCE_LATERAL_DISTANCE,
                 avoidance_lateral_speed_,
                 AVOIDANCE_FORWARD_DISTANCE,
                 avoidance_forward_speed_);
        ROS_INFO("Radar avoidance mode: one-shot per node process");
    }

    bool setEnabledCallback(std_srvs::SetBool::Request &request,
                            std_srvs::SetBool::Response &response)
    {
        if (enabled_ == request.data)
        {
            response.success = true;
            response.message = enabled_ ? "already enabled" : "already disabled";
            return true;
        }

        enabled_ = request.data;
        maneuver_state_ = ManeuverState::IDLE;
        resetAvoidance();
        direction_received_ = false;
        turning_mode_ = false;
        is_stopped_ = false;
        stable_count_ = 0;
        current_error_ = 0.0;
        prev_start_vision_line2_ = 0;
        initPID();
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            vision_data_ = {0.0f, 0.0f, false, false};
        }

        if (!enabled_)
        {
            geometry_msgs::Twist stop;
            cmd_vel_pub_.publish(stop);
        }

        response.success = true;
        response.message = enabled_ ? "enabled" : "disabled";
        ROS_INFO("Vision controller %s", response.message.c_str());
        return true;
    }

    // 数据处理函数
    void processVisionData()
    {
        geometry_msgs::Twist cmd;
        cmd.linear.x = 0.0;
        cmd.linear.y = 0.0;
        cmd.angular.z = 0.0;

        // 检查启动标志
        nh_.getParam("/start_vision1", start_vision_);
        nh_.getParam("/start_vision_line2", start_vision_line2_);

        // DONE 状态：停车等待 /start_vision1 == 1
        if (maneuver_state_ == ManeuverState::DONE)
        {
            if (start_vision_ == 1)
            {
                maneuver_state_ = ManeuverState::IDLE;
                resetAvoidance();
                turning_mode_ = false;
                stable_count_ = 0;
                current_error_ = 0.0;
                ROS_INFO("Maneuver: DONE -> IDLE (start_vision1 == 1)");
            }
            cmd_vel_pub_.publish(cmd);
            return;
        }

        // FORWARD 状态：前进0.35m
        if (maneuver_state_ == ManeuverState::FORWARD)
        {
            cmd.linear.x = 0.3;
            cmd_vel_pub_.publish(cmd);
            if ((ros::Time::now() - maneuver_start_time_).toSec() >= forward_duration_)
            {
                if (needs_rotate_)
                {
                    maneuver_state_ = ManeuverState::ROTATE;
                    maneuver_start_time_ = ros::Time::now();
                    ROS_INFO("Maneuver: FORWARD -> ROTATE");
                }
                else
                {
                    maneuver_state_ = ManeuverState::DONE;
                    std_msgs::String dir_msg;
                    dir_msg.data = last_direction_;
                    direction_pub_.publish(dir_msg);
                    ROS_INFO("Maneuver: FORWARD -> DONE (straight)");
                }
            }
            return;
        }

        // ROTATE 状态：旋转75度
        if (maneuver_state_ == ManeuverState::ROTATE)
        {
            cmd.angular.z = turning_angular_vel_ * maneuver_direction_;
            cmd_vel_pub_.publish(cmd);
            if ((ros::Time::now() - maneuver_start_time_).toSec() >= rotate_duration_)
            {
                maneuver_state_ = ManeuverState::DONE;
                std_msgs::String dir_msg;
                dir_msg.data = last_direction_;
                direction_pub_.publish(dir_msg);
                ROS_INFO("Maneuver: ROTATE -> DONE");
            }
            return;
        }

        if (!start_vision_)
        {
            if (avoidance_state_ != AvoidanceState::IDLE)
            {
                resetAvoidance();
                ROS_INFO("Avoidance cancelled because start_vision1 is disabled");
            }
            turning_mode_ = false;
            is_stopped_ = false;
            stable_count_ = 0;
            current_error_ = 0.0;
            prev_start_vision_line2_ = 0;
            cmd_vel_pub_.publish(cmd);
            return;
        }

        bool start_line2_trigger = (start_vision_line2_ && !prev_start_vision_line2_);
        prev_start_vision_line2_ = start_vision_line2_;

        // if (start_line2_trigger && !turning_mode_)
        // {
        //     turning_mode_ = true;
        //     turn_start_time_ = ros::Time::now();
        //     ROS_INFO("TURNING active: enter fixed rotate mode (%.2fs)", FIXED_TURN_DURATION);
        // }

        // if (turning_mode_)
        // {
        //     if ((ros::Time::now() - turn_start_time_) >= ros::Duration(FIXED_TURN_DURATION))
        //     {
        //         turning_mode_ = false;
        //         stable_count_ = 0;
        //         current_error_ = 0.0;
        //         nh_.setParam("/start_vision_line2", 0);
        //         ROS_INFO("TURNING finished: fixed rotate done, set start_vision_line2=0");
        //     }
        //     else
        //     {
        //         turning_angular_vel_ = nh_.param("/turning_angular_vel", 0.5);
        //         cmd.angular.z = turning_angular_vel_;
        //         cmd_vel_pub_.publish(cmd);
        //         return;
        //     }
        // }

        if (is_stopped_)
        {
            cmd_vel_pub_.publish(cmd);
            return;
        }

        // 雷达避障优先于视觉巡线控制。
        if (processAvoidance(cmd))
        {
            cmd_vel_pub_.publish(cmd);
            return;
        }

        // 读取最新数据（线程安全）
        float x, y;
        bool new_data, valid;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            x = vision_data_.x;
            y = vision_data_.y;
            new_data = vision_data_.new_data;
            valid = vision_data_.valid;
            vision_data_.new_data = false;
        }

        if (!new_data || !valid)
        {
            return;
        }

        // 读取目标y和当前y坐标
        nh_.getParam("/target_y", target_y_);
        current_y_ = nh_.param("CarY", 0.0);
        double y_err = target_y_ - current_y_;

        // 按当前方向选用角速度PID参数（切换时清空积分/微分记忆）
        applyAngularPIDForDir(last_direction_);

        // 处理视觉数据（y在有效范围时计算速度）
        if (y >= Y_LOWER_BOUND && y <= Y_UPPER_BOUND)
        {
            // 计算角速度（基于x误差）
            current_error_ = x - X_REFERENCE;
            double angular_output = -calculatePID(current_error_, angular_pid_);
            cmd.angular.z = angular_output;

            // 计算x方向线速度（基于y误差）
            // if (fabs(y_err) > Y_ERROR_TOLERANCE)
            // {
            //     cmd.linear.x = calculateLinearPID(fabs(y_err));
            //     ROS_INFO("Linear x calculated: %.3fm/s", cmd.linear.x);
            // }
            // else
            // {
            //     cmd.linear.x = 0.0;
            // }
            cmd.linear.x = 0.5; // 恒定速度
            // 发布速度指令
            cmd_vel_pub_.publish(cmd);

            // 定期打印信息
            static ros::Time last_info = ros::Time::now();
            if (ros::Time::now() - last_info > ros::Duration(0.5))
            {
                ROS_INFO("Y error: %.2fm | X error: %.1fpx | Linear: %.3fm/s | Angular: %.3frad/s",
                         y_err, current_error_, cmd.linear.x, cmd.angular.z);
                last_info = ros::Time::now();
            }
        }
        else
        {
            // y值无效时停止运动
            cmd_vel_pub_.publish(cmd);
            ROS_DEBUG("Y value out of range (%.1f), stopping motion", y);
        }
    }

    // 主循环函数
    void run()
    {
        ros::Rate rate(LOOP_RATE);
        while (ros::ok())
        {
            ros::spinOnce();
            if (enabled_)
            {
                processVisionData();
                rate.sleep();
            }
            else
            {
                ros::WallDuration(1.0 / disabled_rate_).sleep();
            }
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vision_x_error_controller");
    VisionErrorController controller;
    controller.run();
    return 0;
}
