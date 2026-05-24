#include <ros/ros.h>
#include "pcl_work/ultrasound.h"
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include "object_information_msgs/Object.h"
#include "std_srvs/Empty.h"
#include "geometry_msgs/Twist.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include "math.h"
#include <actionlib_msgs/GoalID.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>
#include "camera_2d_lidar_calibration/vision_points.h"
#include "ourgoal/getLaserPoint.h"


// 定义物品类别的ID。
#define CMD_Fruits 1170   // 水果
#define CMD_Vegetables 1171   // 蔬菜
#define CMD_sweet 1172 // 甜品

// 定义具体物品ID
// 水果
#define CMD_Apple 530   // 苹果
#define CMD_Banana 609   // 香蕉
#define CMD_Watermelon 1086   // 西瓜

// 蔬菜 
#define CMD_pepper 652   // 辣椒
#define CMD_Tomato 660   // 西红柿
#define CMD_Potato 663    // 土豆

// 甜品 
#define CMD_Milk 429    // 牛奶
#define CMD_Cake 404    // 蛋糕 
#define CMD_coke 418   // 可乐

int cl;
double vision_time;
int hold_flag;
int consecutive_similar_positions_ = 0;
double threshold_distance_;
int min_data_points_;
double min_duration_;
geometry_msgs::Point last_position_;
ros::Time last_update_time_;
double last_yaw_;
int x_id;
int x1, x2;
double k1, k2;
int count_printf = 0;
int target_class;

double LaserPoints_array[3][650];

ourgoal::getLaserPoint vision_srv1;
ourgoal::getLaserPoint vision_srv2;
ourgoal::getLaserPoint vision_srv3;
ourgoal::getLaserPoint vision_srv4;
ros::ServiceClient vision_gettool_client_;

void PrintfArray(double arr[], int size)
{
    // printf("size:%d\n", size);
    for (int i = 0; i < size; i++)
    {
        printf("%lf\n", arr[i]);
    }
}

double calculateDistance(const geometry_msgs::Point &p1, const geometry_msgs::Point &p2)
{
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2) + std::pow(p1.z - p2.z, 2));
}



void ERRORCallback2(const pcl_work::ultrasoundConstPtr &msg)
{
    ros::NodeHandle nh;
    double ErrorX = msg->distance_qian_x;
    double ErrorY = -msg->distance_you_y;
    nh.setParam("ErrorX", ErrorX);
    nh.setParam("ErrorY", ErrorY);
}

void getYoloMsg(const object_information_msgs::ObjectConstPtr &msg)
{
    ros::NodeHandle nh;
    int flag = 0;
    vision_gettool_client_ = nh.serviceClient<ourgoal::getLaserPoint>("/srv_getLaserPoint");
    int res = 0;
    for (int i = 0; i < msg->cl.size(); i++){
        res += (int)(msg->cl[i]);
        }
    cl = res;

    if (nh.param("yolo_begin", 0))
    {
        vision_time = msg->header.stamp.toSec();
        nh.setParam("vision_time", vision_time);
        nh.getParam("target_class", target_class);
        double cx = msg->cx;
        float score = msg -> score;
        x_id = cx;
        nh.setParam("yolo_cx", cx);
        int length = msg->right - msg->left;
        int width = msg->bottom - msg->top;
        x1 = msg->left;
        x2 = msg->right;

        if (target_class == CMD_Fruits)
        {
            if (cl == CMD_Apple || cl == CMD_Banana || cl == CMD_Watermelon)
            {
                flag = 1;
                nh.setParam("cl", cl);
            }
        }
        else if (target_class == CMD_Vegetables)
        {
            if (cl == CMD_pepper || cl == CMD_Tomato || cl == CMD_Potato)
            {
                flag = 1;
                nh.setParam("cl", cl);
            }
        }
        else if (target_class == CMD_sweet)
        {
            if (cl == CMD_Milk || cl == CMD_Cake || cl == CMD_coke)
            {
                flag = 1;
                nh.setParam("cl", cl);
            }
        }
        ROS_WARN("score:%.2f", score);
        

        if (cx > 100 && cx < 540 && flag == 1 && score > 0.7)
        {
            count_printf++;
            nh.setParam("/vision_gettool", 1);
            vision_srv1.request.center_x = x_id;
            vision_srv1.request.left_x = x1;
            vision_srv1.request.right_x = x2;
            vision_srv1.request.mode = 0;

            vision_gettool_client_.call(vision_srv1);

            if (count_printf == 1)
            {
                printf("x1/x2:%d/%d\n", x1, x2);
            }
            nh.setParam("dx1", vision_srv1.response.dx_left);
            nh.setParam("dx2", vision_srv1.response.dx_right);
            nh.setParam("dy1", vision_srv1.response.dy_left);
            nh.setParam("dy2", vision_srv1.response.dy_right);
            nh.setParam("vision_tool_a", vision_srv1.response.line_a);
            nh.setParam("vision_tool_b", vision_srv1.response.line_b);
	    nh.setParam("center_x", vision_srv1.request.center_x);
            ROS_WARN("sb");
            ROS_INFO("dx1:%lf", vision_srv1.response.dx_left);
            ROS_INFO("dx2:%lf", vision_srv1.response.dx_right);
            ROS_INFO("dy1:%lf", vision_srv1.response.dy_left);
            ROS_INFO("dy2:%lf", vision_srv1.response.dy_right);
            ROS_INFO("vision_tool_a:%lf", vision_srv1.response.line_a);
            ROS_INFO("vision_tool_b:%lf", vision_srv1.response.line_b);
	    ROS_INFO("center_x:%d", vision_srv1.request.center_x);
        }
    }

}

void odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
    ros::NodeHandle nh;
    // 位置
    geometry_msgs::Point position = msg->pose.pose.position;
    geometry_msgs::Quaternion orientation_quaternion = msg->pose.pose.orientation;

    double distance = calculateDistance(position, last_position_);
    last_position_ = position;
    // ROS_INFO("distance is %lf", distance);
    if (distance < threshold_distance_)
    {
        // If elapsed time is greater than min_duration, reset counter
        if ((ros::Time::now() - last_update_time_).toSec() > min_duration_)
        {
            consecutive_similar_positions_ = 0;
        }
        consecutive_similar_positions_++;
        last_update_time_ = ros::Time::now();

        // If consecutive similar positions exceed threshold, trigger alert
        if (consecutive_similar_positions_ >= min_data_points_)
        {
            hold_flag = 1;
            // ROS_WARN("The robot may be oscillating in a small distance!");
        }
    }
    else
    {
        hold_flag = 0;
        consecutive_similar_positions_ = 0;
    }

    nh.setParam("hold_flag", hold_flag);
    // ROS_INFO("hold_flag is %d", hold_flag);

    tf::Quaternion tf_quaternion(orientation_quaternion.x, orientation_quaternion.y, orientation_quaternion.z, orientation_quaternion.w);
    double roll, pitch, yaw;
    double CarX, CarY, CarZ, CarW;
    tf::Matrix3x3(tf_quaternion).getRPY(roll, pitch, yaw);

    CarX = msg->pose.pose.position.x;
    CarY = msg->pose.pose.position.y;
    CarZ = msg->pose.pose.orientation.z;
    CarW = msg->pose.pose.orientation.w;
    // Print current orientation (yaw)
    // nh.setParam("CarX", CarX);
    // nh.setParam("CarY", CarY);
    // nh.setParam("CarZ", CarZ);
    // nh.setParam("CarW", CarW);
    nh.setParam("Car_Yaw", yaw);
    // ROS_INFO("Current orientation (yaw): %f", yaw);
    // ROS_INFO("Current position: x = %f, y = %f, z = %f", position.x, position.y, position.z);

    // 判断大转弯
    double angular = msg->twist.twist.angular.z;
    if (fabs(angular) > 1)
    {
        nh.setParam("angular_on", 0);
    }
    else
    {
        nh.setParam("angular_on", 1);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "callback_node");
    ros::NodeHandle nh_;

    nh_.param<double>("threshold_distance", threshold_distance_, 0.05);
    nh_.param<int>("min_data_points", min_data_points_, 10);
    nh_.param<double>("min_duration", min_duration_, 0.001);
    nh_.setParam("vision_gettool", 0);
    nh_.setParam("vision_errX", 0);
    nh_.setParam("vision_errY", 0);
    nh_.setParam("vision_gettool_angle", 0);
    nh_.setParam("dx1", 0);
    nh_.setParam("dx2", 0);
    nh_.setParam("dy1", 0);
    nh_.setParam("dy2", 0);
    nh_.setParam("ultrasound_angle", 0);
    nh_.setParam("distance_error", 0);
    nh_.setParam("center_distance", 0);

    ros::Subscriber sub_odom = nh_.subscribe("/odom", 1, odomCallback);
    ros::Subscriber sub_obj = nh_.subscribe("/Objects", 50, getYoloMsg);
    ros::Subscriber sub_ultrasound2 = nh_.subscribe("/ultrasound/ultra2", 10, ERRORCallback2);

    ros::spin();
    return 0;
}

