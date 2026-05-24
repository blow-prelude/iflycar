#include <ros/ros.h>
#include "geometry_msgs/Twist.h"
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/String.h"
#include <vector>
#include <cmath>
#include <mutex>

class VisionErrorController
{
private:
    // 配置参数
    const double MAX_ANGULAR_VEL = 1.0;    // 最大角速度 (rad/s)
    const double MAX_LINEAR_VEL = 0.5;     // 最大线速度 (m/s)
    const double MIN_LINEAR_VEL = 0.05;    // 最小线速度 (m/s)
    const int PIXEL_ERROR_THRESHOLD = 10;  // x 误差收敛阈值
    const int STABLE_COUNT_THRESHOLD = 5;  // 误差稳定计数阈值
    const int Y_LOWER_BOUND = 340;         // y 值有效范围下限
    const int Y_UPPER_BOUND = 420;         // y 值有效范围上限
    const double X_REFERENCE = 0.0;        // x 方向参考值
    const double Y_ERROR_TOLERANCE = 0.05; // y方向位置误差容忍值 (m)
    const double LOOP_RATE = 50.0;         // 主循环频率 (Hz)

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
    double rotate_duration_;    // (75° * π/180) / angular_vel ≈ 2.618s
    ros::Subscriber direction_sub_;
    ros::Publisher direction_pub_;
    std::string last_direction_; // 存储收到的方向，供转发

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
    ros::Subscriber vision_sub_;
    PIDController angular_pid_; // 角速度PID控制器
    PIDController linear_pid_;  // 线速度PID控制器
    VisionData vision_data_;
    int start_vision_;
    int start_vision_line2_;
    bool turning_mode_;
    bool is_stopped_;
    double current_error_;
    int stable_count_;
    double target_y_;  // 目标y坐标（来自switch.cpp）
    double current_y_; // 当前y坐标（来自参数服务器）
    double turning_angular_vel_;
    std::mutex data_mutex_;

    // 初始化PID参数
    void initPID()
    {
        // 角速度PID参数
        angular_pid_ = {0.01, 0.0, 0.008, 0, 0, 0, MAX_ANGULAR_VEL};
        // 线速度PID参数
        linear_pid_ = {0.5, 0.01, 0.05, 0, 0, 0, MAX_LINEAR_VEL};
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

    // 方向指令回调函数
    void directionCallback(const std_msgs::String::ConstPtr &msg)
    {
        if (maneuver_state_ != ManeuverState::IDLE)
            return;
        std::string dir = msg->data;
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
        else if (dir == "stop")
        {
            last_direction_ = "stop";
            maneuver_state_ = ManeuverState::DONE;
            std_msgs::String dir_msg;
            dir_msg.data = last_direction_;
            direction_pub_.publish(dir_msg);
            ROS_INFO("Maneuver: IDLE -> DONE (stop)");
            return;
        }
        else
        {
            return;
        }
        maneuver_state_ = ManeuverState::FORWARD;
        maneuver_start_time_ = ros::Time::now();
        ROS_INFO("Maneuver: IDLE -> FORWARD (direction=%s)", dir.c_str());
    }

    // 视觉数据回调函数
    void visionCallback(const std_msgs::Float32MultiArray::ConstPtr &msg)
    {
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
    VisionErrorController() : start_vision_(0), start_vision_line2_(0), turning_mode_(false), is_stopped_(false),
                              current_error_(0.0), stable_count_(0)
    {
        // 初始化视觉数据
        vision_data_ = {0.0f, 0.0f, false, false};

        // 创建发布者和订阅者
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
        vision_sub_ = nh_.subscribe("/vision_line", 10, &VisionErrorController::visionCallback, this);
        direction_sub_ = nh_.subscribe("/vision_line_direction", 10, &VisionErrorController::directionCallback, this);
        direction_pub_ = nh_.advertise<std_msgs::String>("/vision_line_direction_out", 10);

        // 读取参数服务器配置
        turning_angular_vel_ = nh_.param("/turning_angular_vel", 0.5);
        forward_duration_ = 0.35 / 0.3;
        rotate_duration_ = (75.0 * M_PI / 180.0) / turning_angular_vel_;
        initPID();

        // 打印初始化信息
        ROS_INFO("Vision Line Controller started (x-vel + angular control)");
        ROS_INFO("Loop rate: %.1fHz | Valid Y range: [%d, %d]",
                 LOOP_RATE, Y_LOWER_BOUND, Y_UPPER_BOUND);
        ROS_INFO("Max linear vel: %.2fm/s | Max angular vel: %.2frad/s",
                 MAX_LINEAR_VEL, MAX_ANGULAR_VEL);
        ROS_INFO("Turning cmd angular: %.2frad/s", turning_angular_vel_);
    }

    // 数据处理函数
    void processVisionData()
    {
        geometry_msgs::Twist cmd;
        cmd.linear.x = 0.0;
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

        // IDLE 状态：原有循线逻辑
        if (!start_vision_)
        {
            turning_mode_ = false;
            is_stopped_ = false;
            stable_count_ = 0;
            current_error_ = 0.0;
            return;
        }

        if (start_vision_line2_)
        {
            if (!turning_mode_)
            {
                turning_mode_ = true;
                ROS_INFO("TURNING active: enter rotate mode");
            }

            turning_angular_vel_ = nh_.param("/turning_angular_vel", 0.5);
            cmd.angular.z = turning_angular_vel_;
            cmd_vel_pub_.publish(cmd);
            return;
        }

        if (turning_mode_)
        {
            turning_mode_ = false;
            stable_count_ = 0;
            current_error_ = 0.0;
            ROS_INFO("TURNING finished: back to line tracking");
        }

        if (is_stopped_)
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
            cmd.linear.x = 0.8; // 恒定速度
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
            processVisionData();
            ros::spinOnce();
            rate.sleep();
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