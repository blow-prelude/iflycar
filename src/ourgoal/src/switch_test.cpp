// // 跑定点
// #include "switch.h"

// // 导航点宏定义
// #define goto_A1 sendPos(1.7, -0.05, -0.0872, 0.9962)
// #define goto_A sendPos(0.85, 0.45, 1.0, 0.0)
// #define goto_B sendPos(0.86, 3.25, 0.707, 0.707)
// #define goto_C sendPos(0.86, 3.25, 0.0, 1.0)
// #define goto_D1 sendPos(3.3, 4.0, 0.707, 0.707)
// #define goto_D2 sendPos(4.2, 4.0, 0.707, 0.707)
// #define goto_E1 sendPos(3.0, 3.4, -0.7071, 0.7071)
// #define goto_E2 sendPos(4.4, 3.5, -0.7071, 0.7071) 
// #define goto_Center sendPos(3.60, 1.2, -0.707, 0.707)
// #define goto_F sendPos(3.70, -0.60, -0.707, 0.707)

// // 物品ID宏定义
// #define CMD_Fruits 1170
// #define CMD_Vegetables 1171
// #define CMD_sweet 1172
// #define CMD_Apple 530
// #define CMD_Banana 609
// #define CMD_Watermelon 1086
// #define CMD_pepper 652
// #define CMD_Tomato 660
// #define CMD_Potato 663
// #define CMD_Milk 429
// #define CMD_Cake 404
// #define CMD_coke 418
// #define CMD_Gazebo1 1
// #define CMD_Gazebo2 2
// #define CMD_Gazebo3 3
// #define CMD_Intersection1 951  // 对应绿色信号灯
// #define CMD_Intersection2 952  // 对应红色信号灯
// #define CMD_OVER ((4 << 3) + 0)

// typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseAction;

// // 构造函数
// OURSWITCH::OURSWITCH()
//     : ac_(nh_, "move_base", true)
// {
//     ROS_INFO("switch node initialized");

//     current_state = GOTOA_;
//     position_srv.request.ask = true;
//     current_color_ = 0;  // 初始化颜色结果（0=未识别）

//     // 初始化视觉服务客户端
//     vision_gettool_client_ = nh_.serviceClient<ourgoal::getLaserPoint>("/srv_getLaserPoint");

//     // 初始化center_x控制的PID参数
//     pid_center_x_.kp = 0.0005;   // 比例系数
//     pid_center_x_.ki = 0.0001;  // 积分系数
//     pid_center_x_.kd = 0.001;   // 微分系数
//     pid_center_x_.err = 0;
//     pid_center_x_.err_last = 0;
//     pid_center_x_.integral = 0;
//     pid_center_x_.output = 0;

//     // 初始化距离控制的PID参数
//     pid_distance_.kp = 0.25;     // 距离控制比例系数
//     pid_distance_.ki = 0.02;    // 距离控制积分系数
//     pid_distance_.kd = 0.01;     // 距离控制微分系数
//     pid_distance_.err = 0;
//     pid_distance_.err_last = 0;
//     pid_distance_.integral = 0;
//     pid_distance_.output = 0;

//     // 初始化发布者
//     cancel_pub = nh_.advertise<actionlib_msgs::GoalID>("move_base/cancel", 10);
//     cmd_vel_pub__ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
//     Gazebo_Command = nh_.advertise<std_msgs::Int32>("class_command", 10);

//     // 初始化订阅者
//     teb_param_reloader = nh_.serviceClient<ourgoal::srv_reload>("/param_reload");
//     getPosition_client = nh_.serviceClient<ourgoal::getPosition>("/srv_getPosition");
//     sub_ultrasound = nh_.subscribe("/ultra", 10, &OURSWITCH::UltrasoundCallback, this);

//     Start_CarX = nh_.param("CarX", 0.0);
//     Start_CarY = nh_.param("CarY", 0.0);

//     ros::service::waitForService("/param_reload");
//     ROS_WARN("Initialization complete!");

//     // 语音启动
//     play_flag_client = nh_.serviceClient<std_srvs::Empty>("/play_flag_srv");
//     ac_.waitForServer(ros::Duration(2));
//     while (!nh_.param("awake", 0));
//     ROS_WARN("System started!");
// }

// // 析构函数
// OURSWITCH::~OURSWITCH()
// {
//     ROS_WARN("switch node terminated");
// }

// // 延迟函数
// void OURSWITCH::delayedFunction(int delayInSeconds)
// {
//     std::this_thread::sleep_for(std::chrono::seconds(delayInSeconds));
// }

// // 打印数组
// void OURSWITCH::PrintfArray(float arr[], int size)
// {
//     printf("Array size: %d\n", size);
//     for (int i = 0; i < size; i++)
//     {
//         printf("Element %d: %f\n", i, arr[i]);
//     }
// }



// // 获取点云处理后的目标点
// void OURSWITCH::getPoint(Point *P)
// {
//     getPosition_client.call(position_srv);
//     fusion_size = position_srv.response.fusion_size;
    
//     if (fusion_size != 0)
//     {
//         P->px = position_srv.response.px;
//         P->py = position_srv.response.py;
//         P->ow = position_srv.response.ow;
//         P->oz = position_srv.response.oz;
//     }
// }

// // 坐标平移
// point_2d OURSWITCH::translate(point_2d p, double dx, double dy)
// {
//     p.x += dx;
//     p.y += dy;
//     return p;
// }

// // 坐标旋转
// point_2d OURSWITCH::rotate(point_2d p, double CarYaw)
// {
//     double rad = CarYaw;
//     double NewPointX = p.x * cos(rad) - p.y * sin(rad);
//     double NewPointY = p.x * sin(rad) + p.y * cos(rad);
//     p.x = NewPointX;
//     p.y = NewPointY;
//     return p;
// }

// // 欧拉角转四元数
// Quaternion OURSWITCH::eulerToQuaternion(double roll, double pitch, double yaw)
// {
//     double cy = cos(yaw * 0.5);
//     double sy = sin(yaw * 0.5);
//     double cp = cos(pitch * 0.5);
//     double sp = sin(pitch * 0.5);
//     double cr = cos(roll * 0.5);
//     double sr = sin(roll * 0.5);

//     Quaternion q;
//     q.w = cr * cp * cy + sr * sp * sy;
//     q.x = sr * cp * cy - cr * sp * sy;
//     q.y = cr * sp * cy + sr * cp * sy;
//     q.z = cr * cp * sy - sr * sp * cy;
//     return q;
// }

// // 发送导航目标点
// void OURSWITCH::sendPos(double x, double y, double z, double w)
// {
//     move_base_msgs::MoveBaseGoal goal;
//     goal.target_pose.header.stamp = ros::Time::now();
//     goal.target_pose.header.frame_id = "map";

//     goal.target_pose.pose.position.x = x;
//     goal.target_pose.pose.position.y = y;
//     goal.target_pose.pose.position.z = 0;
//     goal.target_pose.pose.orientation.x = 0;
//     goal.target_pose.pose.orientation.y = 0;
//     goal.target_pose.pose.orientation.z = z;
//     goal.target_pose.pose.orientation.w = w;

//     ROS_INFO("Sending goal: x=%.2f, y=%.2f, z=%.2f, w=%.2f", x, y, z, w);
//     ac_.sendGoal(goal);
// }


// void OURSWITCH::GotoA()
// {
    
//     ros::Time rotate_start = ros::Time::now();
//     while (ros::Time::now() - rotate_start < ros::Duration(3.5))
//     {
// 	cancel_pub.publish(cancel_msg);
// 	cmd_vel.linear.x = 0.4;
// 	cmd_vel.linear.y = 0.0;
// 	cmd_vel.angular.z = 0.0;  
// 	cmd_vel_pub__.publish(cmd_vel);
//         ros::spinOnce();
//     }
    
//     // 停止移动
//     cmd_vel.linear.x = 0.0;
//     cmd_vel_pub__.publish(cmd_vel);

//     goto_A;
//     ac_.waitForResult();
//     while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
//         ros::spinOnce();
    
//     goto_B;
//     ac_.waitForResult();
//     while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
//         ros::spinOnce();
        
//     goto_C;
//     ac_.waitForResult();
//     while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
//         ros::spinOnce();

//     goto_D1;
//     ac_.waitForResult();
//     while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
//         ros::spinOnce();

//     goto_D2;
//     ac_.waitForResult();
//     while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
//         ros::spinOnce();
//     goto_E1;
//     ac_.waitForResult();
//     while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
//         ros::spinOnce();

//     goto_E2;
//     ac_.waitForResult();
//     while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
//         ros::spinOnce();
    
//     goto_Center;
//     ac_.waitForResult();
//     while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
//         ros::spinOnce();

//     goto_F;
//     ac_.waitForResult();
//     while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
//         ros::spinOnce();
// }

// // 二维码回调函数
// void OURSWITCH::getZbarCallback(const std_msgs::String::ConstPtr &msg)
// {
//     std::string received_msg = msg->data;
//     if(Zbar_flag == 1)
//     {
//         if (received_msg == "Fruit") {
//         nh_.setParam("target_class", CMD_Fruits);
//         Zbar_flag = 0;
//         // ROS_INFO("Received: Fruit");
//         } 
//         else if (received_msg == "Vegetable") {
//             nh_.setParam("target_class", CMD_Vegetables);
//             Zbar_flag = 0;
//             // ROS_INFO("Received: Vegetable");
//         } 
//         else if (received_msg == "Dessert") {
//             nh_.setParam("target_class", CMD_sweet);
//             Zbar_flag = 0;
//             // ROS_INFO("Received: Dessert");
//         } 
//         else {
//             ROS_WARN("Received unknown message: %s", received_msg.c_str());
//         }
//     }
    
// }

// // 视觉角度回调
// void OURSWITCH::visionAngleCallback(const std_msgs::Float32MultiArray::ConstPtr& msg)
// {
//     if (msg->data.size() >= 3)
//     {
//         vision_X = msg->data[0];
//         vision_Y = msg->data[1];
//         vision_Angle = msg->data[2];
//     }
//     else
//     {
//         ROS_WARN("Incomplete vision angle data (size: %zu)", msg->data.size());
//     }
// }

// // 仿真房间号回调
// void OURSWITCH::getGzeboRoomCallback(const std_msgs::Int32::ConstPtr &msg) 
// {
//     if(gazebo_flag = 1)
//     {
//         target = msg->data;
//     }
    
// }

// // 仿真食物号回调
// void OURSWITCH::getGzeboFoodCallback(const std_msgs::Int32::ConstPtr &msg) 
// {
//     target2 = msg->data;
//     // ROS_INFO("Received food number: %d", target2);
// }

// // 超声波数据回调
// void OURSWITCH::UltrasoundCallback(const pcl_work::ultrasoundConstPtr &msg)
// {
//     distance_qian_x = msg->distance_qian_x; 
//     distance_zuo_y = msg->distance_zuo_y;
//     distance_hou_x = -msg->distance_hou_x;
//     distance_you_y = -msg->distance_you_y;
// }

// // 新增：颜色识别结果回调函数
// void OURSWITCH::colorResultCallback(const std_msgs::Int32::ConstPtr& msg)
// {
//     current_color_ = msg->data;
//     // ROS_INFO("Received color result: %d (1=green, 2=red)", current_color_);
    
//     // 验证结果有效性
//     if (current_color_ != 1 && current_color_ != 2)
//     {
//         // ROS_WARN("Invalid color value: %d (expected 1 or 2)", current_color_);
//         current_color_ = 0; // 标记为无效
//     }
// }
// // 主函数
// int main(int argc, char **argv)
// {
//     ros::init(argc, argv, "switch_node");
//     ros::NodeHandle nh;
    
//     OURSWITCH ucar;

//     ucar.GotoA();
    

//     return 0;
// }
    


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