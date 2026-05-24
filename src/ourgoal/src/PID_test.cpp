#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <pcl_work/ultrasound.h>

class OURSWITCH
{
private:
    ros::NodeHandle nh_;
    ros::Publisher cmd_pub_;
    ros::Subscriber sub_ultrasound_;
    ros::Subscriber sub_odom_;

    // 雷达四个正方向的距离
    double distance_qian_x;
    double distance_zuo_y;
    double distance_hou_x;
    double distance_you_y;
    
    // 车头偏航角
    double yaw_;

    // 各个方向独立的设置距离
    double safe_F, safe_B, safe_L, safe_R;
    
    // PID 增益
    double Kp_dist = 1.0;  // 距离控制增益
    double Kp_yaw = 2.0;   // 航向角锁定增益
    double max_vel = 0.5;  // 最大限速

public:
    OURSWITCH() : distance_qian_x(9.9), distance_zuo_y(9.9), distance_hou_x(9.9), distance_you_y(9.9), yaw_(0.0)
    {
        cmd_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
        sub_ultrasound_ = nh_.subscribe("/ultra", 10, &OURSWITCH::UltrasoundCallback, this);
        sub_odom_ = nh_.subscribe("/odom", 10, &OURSWITCH::OdomCallback, this);

        // 初始化各个方向的独立安全距离
        safe_F = 0.10;
        safe_B = 0.35;
        safe_L = 0.20;
        safe_R = 0.20;
    }

    void UltrasoundCallback(const pcl_work::ultrasoundConstPtr &msg)
    {
        distance_qian_x = msg->distance_qian_x; 
        distance_zuo_y = msg->distance_zuo_y;
        distance_hou_x = -msg->distance_hou_x; // 假设负值已处理为绝对距离
        distance_you_y = -msg->distance_you_y;
    }

    void OdomCallback(const nav_msgs::Odometry::ConstPtr &msg)
    {
        tf2::Quaternion q(
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z,
            msg->pose.pose.orientation.w);
        tf2::Matrix3x3 m(q);
        double r, p;
        m.getRPY(r, p, yaw_);
    }

    // 辅助限速函数
    double limit(double val) {
        if (val > max_vel) return max_vel;
        if (val < -max_vel) return -max_vel;
        return val;
    }

    void run()
    {
        ros::Rate rate(20);
        
        // 1. 阅读参数服务器的 start 参数，阻塞直到 start 为 1
        int start_val = 0;
        while (ros::ok() && start_val != 1)
        {
            nh_.getParam("start", start_val);
            ros::spinOnce();
            rate.sleep();
        }

        int escape_state = 1;
        bool task_finished = false;

        while (ros::ok() && !task_finished)
        {
            ros::spinOnce();
            geometry_msgs::Twist cmd;

            // 注意：每个阶段都保证 yaw 角为 0
            cmd.angular.z = Kp_yaw * (0.0 - yaw_);

            double error = 0;

            switch (escape_state)
            {
                case 1: // 向前 (+X)
                    error = distance_qian_x - safe_F;
                    cmd.linear.x = limit(Kp_dist * error);
                    cmd.linear.y = 0;
                    if (std::abs(error) < 0.05) escape_state = 2;
                    break;

                case 2: // 向右 (-Y)
                    error = distance_you_y - safe_R;
                    cmd.linear.y = -limit(Kp_dist * error);
                    cmd.linear.x = 0;
                    if (std::abs(error) < 0.05) escape_state = 3;
                    break;

                case 3: // 向前 (+X)
                    error = distance_qian_x - safe_F;
                    cmd.linear.x = limit(Kp_dist * error);
                    cmd.linear.y = 0;
                    if (std::abs(error) < 0.05) escape_state = 4;
                    break;

                case 4: // 向左 (+Y)
                    error = distance_zuo_y - safe_L;
                    cmd.linear.y = limit(Kp_dist * error);
                    cmd.linear.x = 0;
                    if (std::abs(error) < 0.05) escape_state = 5;
                    break;

                case 5: // 向前 (+X)
                    error = distance_qian_x - safe_F;
                    cmd.linear.x = limit(Kp_dist * error);
                    cmd.linear.y = 0;
                    if (std::abs(error) < 0.05) escape_state = 6;
                    break;

                case 6: // 向右 (-Y)
                    error = distance_you_y - safe_R;
                    cmd.linear.y = -limit(Kp_dist * error);
                    cmd.linear.x = 0;
                    if (std::abs(error) < 0.05) escape_state = 7;
                    break;

                case 7: // 向后 (-X)
                    error = distance_hou_x - safe_B;
                    cmd.linear.x = -limit(Kp_dist * error);
                    cmd.linear.y = 0;
                    if (std::abs(error) < 0.05) escape_state = 8;
                    break;

                case 8: // 向左 (+Y)
                    error = distance_zuo_y - safe_L;
                    cmd.linear.y = limit(Kp_dist * error);
                    cmd.linear.x = 0;
                    if (std::abs(error) < 0.05) escape_state = 9;
                    break;

                case 9: // 向后 (-X)
                    error = distance_hou_x - safe_B;
                    cmd.linear.x = -limit(Kp_dist * error);
                    cmd.linear.y = 0;
                    if (std::abs(error) < 0.05) escape_state = 10;
                    break;

                case 10: // 向右 (-Y)
                    error = distance_you_y - safe_R;
                    cmd.linear.y = -limit(Kp_dist * error);
                    cmd.linear.x = 0;
                    if (std::abs(error) < 0.05) escape_state = 11;
                    break;

                case 11: // 向后 (-X)
                    error = distance_hou_x - safe_B;
                    cmd.linear.x = -limit(Kp_dist * error);
                    cmd.linear.y = 0;
                    if (std::abs(error) < 0.05) {
                        task_finished = true;
                        cmd.linear.x = 0;
                        cmd.linear.y = 0;
                    }
                    break;
            }

            cmd_pub_.publish(cmd);
            rate.sleep();
        }

        // 停止小车
        geometry_msgs::Twist stop;
        cmd_pub_.publish(stop);
        ROS_INFO("任务结束。");
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "warehouse_escape_node");
    OURSWITCH node;
    node.run();
    return 0;
}