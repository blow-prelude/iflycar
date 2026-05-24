#ifndef SWITCH_H
#define SWITCH_H

#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include "geometry_msgs/PoseStamped.h"
#include "std_srvs/Empty.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include "math.h"
#include <actionlib_msgs/GoalID.h>
#include <thread>
#include <chrono>
#include "ourgoal/srv_reload.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "ourgoal/getPosition.h"
#include "std_msgs/String.h"
#include "std_msgs/Int32.h"
#include "std_msgs/Float32MultiArray.h"
#include "ourgoal/getLaserPoint.h"
#include "pcl_work/ultrasound.h"


// 坐标点结构体
typedef struct Point
{
    double px;
    double py;
    double oz;
    double ow;
} Point;

// PID结构体（使用output成员）
typedef struct PID
{
    double kp, ki, kd;   // 三个系数
    float err, err_last; // 误差、上次误差
    float integral;      // 积分
    float output;        // 输出
} PID;

// 二维坐标结构体
typedef struct point_2d
{
    double x;
    double y;
} point_2d;

// 四元数结构体
struct Quaternion
{
    double w, x, y, z;
};

// 任务阶段枚举
enum 
{
    TEST_,      // 测试阶段
    GOTOA_,    // 到达领取采购任务区
    GOTOB_,    // 到达拣货区路口
    VISION_GETTOOL_,   // 视觉获取工具
    GOTOC_,    // 到达仿真区
    Gazebo_,   // 仿真阶段
    GOTOD_,    // 到达1号标牌前
    VISION_LINE_,    // 视觉巡线
    GOTOF_,     // 到达终点（含避障）
    END_,      // 结束阶段
};


class OURSWITCH
{
    public:
        OURSWITCH();       
	    virtual ~OURSWITCH();

        int current_state;

        void delayedFunction(int delayInSeconds);  // 延迟指定秒数
        void PrintfArray(float arr[], int size);  // 打印数组
        double Limit_Value(double INPUT, double MAX, double MIN);    // 限制输出值
        double PID_Realize(PID *pid, double err, double MAX, double MIN);
        double PID_Realize2(PID *pid, double err, double MAX, double MIN, double integral_limit);
        void InitPID();  // PID初始化

        void getPoint(Point *P);    // 获取点云处理后的目标点坐标
        point_2d translate(point_2d p, double dx, double dy);  // 坐标平移
        point_2d rotate(point_2d p, double CarYaw);  // 坐标旋转
        Quaternion eulerToQuaternion(double roll, double pitch, double yaw);  // 欧拉角转四元数

        // 导航与任务函数
        void sendPos(double x, double y, double z, double w);
        void GotoA();
        void GotoB();
        void Vision_GetTool();
        void GotoC();
        void Gazebo();
        void GotoD();
        void vision_line();
        void GotoF();
        void END();

        // 回调函数
        void getZbarCallback(const std_msgs::String::ConstPtr &msg);
        void getGzeboRoomCallback(const std_msgs::Int32::ConstPtr &msg);
        void getGzeboFoodCallback(const std_msgs::Int32::ConstPtr &msg);
        void visionAngleCallback(const std_msgs::Float32MultiArray::ConstPtr& msg);
        void UltrasoundCallback(const pcl_work::ultrasoundConstPtr &msg);
        // 新增：颜色识别结果回调
        void colorResultCallback(const std_msgs::Int32::ConstPtr& msg);
        
    private:
        // 从参数服务器获取center_x
        bool getCenterXFromParam();
        // 计算y方向速度（用于center_x调节）
        double calculateYVelocity();
        // 计算x方向速度（用于距离调节）
        double calculateXVelocity();

        ros::NodeHandle nh_;
        actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> ac_;
        
        // 服务与发布订阅
        ros::ServiceClient vision_gettool_client_;
        ros::Publisher cancel_pub;
        ros::Publisher cmd_vel_pub__;
        ros::Publisher Gazebo_Command;
        ros::Subscriber Gazebo_Room;      
        ros::Subscriber Gazebo_Food;
        ros::Subscriber vision_angle;
	ros::Subscriber sub_ultrasound;
        ros::ServiceClient teb_param_reloader;
        ros::ServiceClient getPosition_client;
        ros::Subscriber vision_zbar_sub;
        ros::ServiceClient play_flag_client;
        // 新增：订阅颜色识别结果话题
        ros::Subscriber color_result_sub;
        ros::Subscriber imu_sub;
        
        // 消息与参数
        geometry_msgs::Twist cmd_vel;
        actionlib_msgs::GoalID cancel_msg;
        ourgoal::getPosition position_srv;
        std_srvs::Empty _;
        ourgoal::srv_reload teb_reload;

        // 任务参数
        int tool = 0, target;
        int gazebo_flag = 1;
        int Zbar_flag = 1;
        int target_class;
        int target1;
        int target2;
        int target3;
        int start_vision_line2;
        double roll, pitch, yaw;
        float vision_X, vision_Y, vision_Angle;
        double vision_y_err;
        double x_adjust_vel;

        // 传感器数据
        double distance_qian_x;
        double distance_hou_x;
        double distance_you_y;
        double distance_zuo_y;

        double Start_CarX;
        double Start_CarY;

        // 状态标志
        int vision_gettool = 0;
        int vision_getIntersection1 = 0;  // 绿色信号灯（1）
        int vision_getIntersection2 = 0;  // 红色信号灯（2）
        double targetX, targetY;
        int fusion_size = 0;
        int Point_count = 0;
        bool vision_control_active_;
        bool vision_control_active2_ = false;  // 新增：第二个视觉控制标志
        // 新增：存储颜色识别结果（1=绿色，2=红色）
        int current_color_;

        // PID与坐标相关
        Point P;
        PID Aid_X;
        PID Aid_Y;
        PID pid_center_x_;  // 用于center_x调节的PID
        PID pid_distance_;  // 用于距离调节的PID
        const double target_center_x_ = 320.0;  // 目标中心x值（固定为320）
        const double target_distance_ = 0.1;    // 目标前方距离（20cm）
        double current_center_x_;  // 当前中心x值（从参数服务器读取）
};

#endif 
    
