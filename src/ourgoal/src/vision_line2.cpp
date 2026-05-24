#include <ros/ros.h>
#include "geometry_msgs/Twist.h"
#include "std_msgs/Float32MultiArray.h"
#include "pcl_work/ultrasound.h"  // Ultrasonic message type
#include <vector>
#include <cmath>
#include <mutex>

class VisionErrorController {
private:
    // Configuration parameters
    const double MAX_ANGULAR_VEL = 0.5;       // Maximum angular velocity (rad/s)
    const double MAX_LINEAR_VEL = 0.3;        // Maximum linear velocity (m/s)
    const double MIN_LINEAR_VEL = 0.05;       // Minimum linear velocity (m/s)
    const int Y_LOWER_BOUND = 340;            // Lower bound of valid y range
    const int Y_UPPER_BOUND = 370;            // Upper bound of valid y range
    const double X_REFERENCE = 0.0;           // X direction reference (vision center)
    const double FRONT_TOLERANCE = 0.08;      // Front distance error tolerance (m)
    const double LOOP_RATE = 50.0;            // Main loop frequency (Hz)

    // PID Controller structure
    struct PIDController {
        double kp, ki, kd;
        double err, err_last, err_sum;
        double output_limit;
    };

    // Vision data structure
    struct VisionData {
        float x;          // X error
        float y;          // Y pixel coordinate
        bool new_data;    // New data flag
        bool valid;       // Data validity flag
    };

    // Class member variables
    ros::NodeHandle nh_;                      // ROS node handle
    ros::Publisher cmd_vel_pub_;              // Velocity command publisher
    ros::Subscriber vision_sub_;              // Vision data subscriber
    ros::Subscriber ultrasound_sub_;          // Ultrasonic data subscriber
    PIDController angular_pid_;               // Angular velocity PID controller
    PIDController front_pid_;                 // Front distance PID controller
    VisionData vision_data_;                  // Vision data storage
    int start_vision_;                        // Start flag (corresponds to start_vision2)
    bool is_stopped_;                         // Stopped state flag
    double current_error_;                    // Current X direction vision error
    double target_front_dist_ = 0.30;                // Target front distance
    double current_front_dist_;               // Current front distance
    std::mutex data_mutex_;                   // Mutex for data access synchronization

    // Initialize PID parameters
    void initPID() {
        // Angular velocity PID parameters
        angular_pid_ = {0.003, 0.0, 0.008, 0, 0, 0, MAX_ANGULAR_VEL};
        
        // Front distance PID parameters
        front_pid_ = {0.5, 0.01, 0.05, 0, 0, 0, MAX_LINEAR_VEL};
    }

    // Generic PID calculation function
    double calculatePID(double err, PIDController& pid, const std::string& pid_name) {
        pid.err = err;
        
        // Accumulate integral term only for small errors
        if (fabs(pid.err) < 20.0) {
            pid.err_sum += pid.err;
            pid.err_sum = std::max(std::min(pid.err_sum, 50.0), -50.0);  // Integral clamping
        } else {
            pid.err_sum = 0;  // Reset integral for large errors
        }
        
        double diff = pid.err - pid.err_last;  // Derivative term
        
        double output = pid.kp * pid.err + pid.ki * pid.err_sum + pid.kd * diff;
        output = std::max(std::min(output, pid.output_limit), -pid.output_limit);  // Output clamping
        
        pid.err_last = pid.err;
        return output;
    }

    // Calculate front distance PID (controls X direction velocity)
    double calculateFrontPID(double err) {
        front_pid_.err = err;
        
        // Integral clamping
        if (fabs(front_pid_.err) < 0.1) {  // Accumulate integral for small errors
            front_pid_.err_sum += front_pid_.err;
            front_pid_.err_sum = std::max(std::min(front_pid_.err_sum, 0.5), -0.5);
        } else {
            front_pid_.err_sum = 0;
        }
        
        // Derivative term
        double diff = front_pid_.err - front_pid_.err_last;
        
        double output = front_pid_.kp * front_pid_.err + 
                       front_pid_.ki * front_pid_.err_sum + 
                       front_pid_.kd * diff;
        
        // Output clamping (ensure minimum velocity)
        if (fabs(output) < MIN_LINEAR_VEL && fabs(err) > FRONT_TOLERANCE) {
            output = (output > 0) ? MIN_LINEAR_VEL : -MIN_LINEAR_VEL;
        }
        output = std::max(std::min(output, front_pid_.output_limit), -front_pid_.output_limit);
        
        front_pid_.err_last = front_pid_.err;
        return output;
    }


    // Vision data callback function
    void visionCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        // Check data format
        if (msg->data.size() != 2) {
            vision_data_.valid = false;
            vision_data_.new_data = true;
            static ros::Time last_warn = ros::Time::now();
            if (ros::Time::now() - last_warn > ros::Duration(1.0)) {
                ROS_WARN("Invalid vision data format (expected 2 elements, got %zu)", msg->data.size());
                last_warn = ros::Time::now();
            }
            return;
        }

        // Store new data
        vision_data_.x = msg->data[0];
        vision_data_.y = msg->data[1];
        vision_data_.valid = true;
        vision_data_.new_data = true;
    }

    // Ultrasonic data callback function
    void ultrasoundCallback(const pcl_work::ultrasoundConstPtr &msg) {
        current_front_dist_ = msg->distance_qian_x;
        ROS_INFO("Received ultrasonic data - front distance: %.3f m", current_front_dist_);
    }

public:
    // Constructor
    VisionErrorController() : start_vision_(0), is_stopped_(false), 
                             current_error_(0.0), current_front_dist_(0.0) {
        // Initialize vision data
        vision_data_ = {0.0f, 0.0f, false, false};
        
        // Create publishers and subscribers
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
        vision_sub_ = nh_.subscribe("/vision_line", 10, &VisionErrorController::visionCallback, this);
        ultrasound_sub_ = nh_.subscribe("/ultra", 10, &VisionErrorController::ultrasoundCallback, this);
        
        // Initialize parameter server
        nh_.setParam("/start_vision2", 0);
        initPID();
        
    }

    // Data processing function
    void processVisionData() {
        geometry_msgs::Twist cmd;
        cmd.linear.x = 0.0;
        cmd.angular.z = 0.0;

        // 检查启动标志
        nh_.getParam("/start_vision2", start_vision_);
        if (!start_vision_) {
            is_stopped_ = false;
            current_error_ = 0.0;
            return;
        }

        if (is_stopped_) {
            cmd_vel_pub_.publish(cmd);
            return;
        }

        // 读取视觉数据（线程安全）
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

        // 计算前端距离误差（无论y是否有效，都处理线速度）
        double front_err = target_front_dist_ - current_front_dist_;
        ROS_INFO("Target: %.3f m | Current: %.3f m | Error: %.3f m",
                target_front_dist_, current_front_dist_, front_err);

        // 线速度控制（仅依赖距离误差，与y无关）
        if (fabs(front_err) > FRONT_TOLERANCE) {
            cmd.linear.x = -calculateFrontPID(front_err);  // 始终调整距离
        } else {
            is_stopped_ = true;
            ROS_INFO("Front error within tolerance - stopping linear motion");
        }

        // 角速度控制（仅当y有效时调整，否则保持0）
        if (new_data && valid && (y >= Y_LOWER_BOUND && y <= Y_UPPER_BOUND)) {
            current_error_ = x - X_REFERENCE;
            cmd.angular.z = -calculatePID(current_error_, angular_pid_, "Angular");
            ROS_INFO("Angular vel: %.3f rad/s (x error: %.1f px)", cmd.angular.z, current_error_);
        } else {
            // y无效时不调整角度（保持当前方向，角速度设为0）
            cmd.angular.z = 0.0;
            if (new_data && !valid) {
                ROS_WARN("Invalid vision data - stopping angular adjustment");
            } else if (new_data && (y < Y_LOWER_BOUND || y > Y_UPPER_BOUND)) {
                ROS_INFO("Y value (%.1f) out of range - stopping angular adjustment", y);
            }
        }

        // 发布速度指令（线速度和角速度独立控制）
        cmd_vel_pub_.publish(cmd);
        ROS_INFO("Published: linear=%.3f m/s, angular=%.3f rad/s", cmd.linear.x, cmd.angular.z);

        // 定期打印综合信息
        static ros::Time last_info = ros::Time::now();
        if (ros::Time::now() - last_info > ros::Duration(0.5)) {
            ROS_INFO("Front: %.3fm | Err: %.3fm | X err: %.1fpx | Linear: %.3f | Angular: %.3f",
                    current_front_dist_, front_err, current_error_, cmd.linear.x, cmd.angular.z);
            last_info = ros::Time::now();
        }
    }

    // Main loop function
    void run() {
        ros::Rate rate(LOOP_RATE);
        ROS_INFO("Entering main control loop with frequency: %.1f Hz", LOOP_RATE);
        
        while (ros::ok()) {
            processVisionData();
            ros::spinOnce();
            rate.sleep();
        }
    }
};

int main(int argc, char**argv) {
    ros::init(argc, argv, "vision_x_error_controller2");
    VisionErrorController controller;
    controller.run();
    return 0;
}