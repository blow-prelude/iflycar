

// 通过服务的方式传坐标和欧拉角
#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include "std_srvs/Empty.h"
#include "geometry_msgs/Twist.h"
#include "math.h"
#include <actionlib_msgs/GoalID.h>
#include <tf2/LinearMath/Quaternion.h>  // 用于四元数转换
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>  // 用于四元数转换

#include "ourgoal/setgoal.h"

#define PI 3.1415926

// 全局变量
double goal_x = 0.0;
double goal_y = 0.0;
double goal_yaw = 0.0;
bool new_goal_received = false;  // 标志位，表示是否有新的目标点

// 定义导航目标点的宏，调用sendPos函数发送目标点。
typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

// 发送导航点
void sendPos(MoveBaseClient *ac, double x, double y, double yaw)
{
    move_base_msgs::MoveBaseGoal goal;
    goal.target_pose.header.stamp = ros::Time::now();
    goal.target_pose.header.frame_id = "map";

    // 设置目标点的位置
    goal.target_pose.pose.position.x = x;
    goal.target_pose.pose.position.y = y;
    goal.target_pose.pose.position.z = 0;

    // 将欧拉角转换为四元数
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);  // 绕 Z 轴旋转 yaw 弧度
    goal.target_pose.pose.orientation = tf2::toMsg(q);  // 转换为 geometry_msgs::Quaternion

    ac->sendGoal(goal);
    ROS_INFO("Sent goal: x=%f, y=%f, yaw=%f", x, y, yaw);
}

// 服务回调函数
bool handleSetGoalRequest(ourgoal::setgoal::Request &req,
                          ourgoal::setgoal::Response &res)
{
    // 保存接收到的目标点
    goal_x = req.x;
    goal_y = req.y;
    goal_yaw = req.yaw;

    // 设置标志位
    new_goal_received = true;

    ROS_INFO("Received new goal: x=%f, y=%f, yaw=%f", goal_x, goal_y, goal_yaw);
    res.success = true;  // 返回成功
    return true;
}

int main(int argc, char **argv)
{
    // 初始化 ROS 节点
    ros::init(argc, argv, "goal_service_node");
    ros::NodeHandle nh;

    // 创建服务端
    ros::ServiceServer service = nh.advertiseService("set_goal", handleSetGoalRequest);
    ROS_INFO("Ready to receive goal requests.");

    // 创建 move_base 客户端
    MoveBaseClient ac("move_base", true);
    ac.waitForServer(ros::Duration(5.0));  // 等待服务器连接

    if (!ac.isServerConnected())
    {
        ROS_ERROR("move_base action server is not connected!");
        return 1;
    }

    // 主循环
    ros::Rate rate(10);  // 10Hz
    while (ros::ok())
    {
        // 检查是否有新的目标点
        if (new_goal_received)
        {
            // 发送目标点
            sendPos(&ac, goal_x, goal_y, goal_yaw);

            // 重置标志位
            new_goal_received = false;

            // 等待结果
            ac.waitForResult(ros::Duration(30.0));  // 最多等待 30 秒

            if (ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
            {
                ROS_INFO("Goal reached successfully!");
            }
            else
            {
                ROS_ERROR("Failed to reach the goal!");
            }
        }

        // 处理回调函数
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}



// #include <ros/ros.h>
// #include <move_base_msgs/MoveBaseAction.h>
// #include <actionlib/client/simple_action_client.h>
// #include "geometry_msgs/PoseStamped.h"
// #include "std_srvs/Empty.h"
// #include "geometry_msgs/Twist.h"
// #include "geometry_msgs/PoseWithCovarianceStamped.h"
// #include <tf2/LinearMath/Quaternion.h>
// #include <tf2/LinearMath/Matrix3x3.h>
// #include "math.h"
// #include <actionlib_msgs/GoalID.h>
// #include <thread>
// #include <chrono>
// #include "ourgoal/srv_reload.h"
// #include "geometry_msgs/PoseWithCovarianceStamped.h"
// #include "ourgoal/getPosition.h"
// #include "ourgoal/srv_reload.h"


// ros::ServiceClient play_flag_client;  // 启动语音播放。
// typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseAction;
// std_srvs::Empty _;
// int target;


// int main(int argc, char **argv)
// {
//     ros::init(argc, argv, "switch_node");
//     ros::NodeHandle nh;


//     ROS_WARN("init ok!!!");

//     MoveBaseAction ac(nh, "move_base", true);
//     // 语音启动
//     play_flag_client = nh.serviceClient<std_srvs::Empty>("/play_flag_srv");
//     ac.waitForServer(ros::Duration(5)); // 语音唤醒延迟

//     while (!nh.param("awake", 0))
//         ;

//     target = 520;
//     nh.setParam("target_tool", target);     // 语音播报会接受target_tool 来进行语音播报

//     nh.setParam("audio", 0); // 语音运行中
//     play_flag_client.call(_);
//     while (!nh.param("audio", 0))

//     // ros::Rate loop_rate(10);

//     // while (ros::ok())
//     // {
    

//     //     loop_rate.sleep();
//     // }
//     return 0;
// }