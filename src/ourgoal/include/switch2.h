#ifndef SWITCH_H
#define SWITCH_H

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
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
#include "sensor_msgs/LaserScan.h"
#include "ourgoal/getLaserPoint.h"
#include "pcl_work/ultrasound.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>  // 用于四元数转换v
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <mutex>


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
    GOTOA_,     // 仓储区迷宫 (利用雷达和PID平移)
    GOTOB_,     // 物品领取区
    XingHuoAI_, // 视觉获取工具
    GOTOC1_,    // 到达实体要求的生产区，进行语音播报
    GOTOC2_,    // 到达仿真要求的生产区，进行仿真任务
    Gazebo_,    // 仿真阶段
    GOTOD_,     // 到达巡线起点
    VISION_LINE_,    // 视觉巡线
    END_,       // 结束阶段
};


class OURSWITCH
{
    public:
        OURSWITCH();       
	    virtual ~OURSWITCH();

        int current_state;
        
        // 三个二维码识别到的物品
        std::string qr_code_1_;
        std::string qr_code_2_;
        std::string qr_code_3_;

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
        void sendPos(double x, double y, double yaw);
        void GotoA();
        void GotoB();
        void XingHuoAI();
        void GotoC(int target_num);  // 原代码头文件写的是 GotoC()，按照cpp内容应为 GotoC(int)
        void Gazebo();
        void GotoD();
        void vision_line();
        void GotoF();
        void END();

        // 回调函数
        void getZbarCallback(const std_msgs::String::ConstPtr &msg);
        void UltrasoundCallback(const pcl_work::ultrasoundConstPtr &msg);
        void OdomCallback(const nav_msgs::Odometry::ConstPtr &msg); // 用于提取高精度yaw
        void ScanCallback(const sensor_msgs::LaserScan::ConstPtr &msg);

        void SignalClassCallback(const std_msgs::Int32::ConstPtr &msg);
        void SignalDetectionCallback(const std_msgs::Float32MultiArray::ConstPtr &msg);
        
    private:
        // 从参数服务器获取center_x
        bool getCenterXFromParam();
        // 计算y方向速度（用于center_x调节）
        double calculateYVelocity();
        bool detectGap(const sensor_msgs::LaserScan &scan,
                       double &mid_x, double &mid_y, double &width) const;

        ros::NodeHandle nh_;
        actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> ac_;
        
        // 服务与发布订阅
        ros::ServiceClient vision_gettool_client_;
        ros::Publisher cancel_pub;
        ros::Publisher cmd_vel_pub__;
        ros::Publisher Gazebo_Command;
        ros::Subscriber sub_ultrasound;
        ros::Subscriber vision_zbar_sub;
        ros::Subscriber sub_odom_; // 订阅底盘odom获取偏航角
        ros::Subscriber sub_scan_;
        ros::ServiceClient teb_param_reloader;
        ros::ServiceClient getPosition_client;
        ros::ServiceClient play_flag_client;

        ros::Subscriber sub_signal_class_;
        ros::Subscriber sub_signal_detection_;
        
        // 消息与参数
        geometry_msgs::Twist cmd_vel;
        actionlib_msgs::GoalID cancel_msg;
        ourgoal::getPosition position_srv;
        std_srvs::Empty _;
        ourgoal::srv_reload teb_reload;

        int current_signal_class_ = -1;
        ros::Time last_signal_class_time_;

        double signal_center_x_ = -1.0;
        double signal_center_y_ = -1.0;
        double signal_box_x_l_ = -1.0;
        double signal_box_x_r_ = -1.0;
        ros::Time last_signal_detection_time_;

        bool target_locked_ = false;
        double locked_center_x_ = -1.0;
        double locked_box_x_l_ = -1.0;
        double locked_box_x_r_ = -1.0;
        int locked_signal_class_ = -1;

        struct WarehouseParkingCache
        {
            bool valid = false;
            double parking_x = 0.0;
            double parking_y = 0.0;
            double parking_yaw = 0.0;
            double observation_x = 0.0;
            double observation_y = 0.0;
            double observation_yaw = 0.0;
        };

        // 食品、日用品、电子产品三个车间在本次运行中的停车位置。
        WarehouseParkingCache warehouse_parking_cache_[3];

        // 只记录“在哪个观察姿态看到了某类别”，不把它当作停车点。
        // 下次以该类别为目标时，GotoC 会先到这里复核，失败后再恢复完整搜索。
        struct WarehouseObservationCache
        {
            bool valid = false;
            double observation_x = 0.0;
            double observation_y = 0.0;
            double observation_yaw = 0.0;
            int wall = -1;
            int first_slot = -1;
            int last_slot = -1;
        };

        WarehouseObservationCache warehouse_observation_cache_[3];

        // 任务参数
        double roll, pitch, yaw;

        // 传感器数据
        double distance_qian_x;
        double distance_hou_x;
        double distance_you_y;
        double distance_zuo_y;

        // PID 和 雷达平移导航参数
        double Kp_dist;
        double Kp_yaw;
        double max_vel;
        double safe_F, safe_B, safe_L, safe_R, safe_R2;

        // GotoD 缺口中点的雷达修正
        std::mutex gap_mutex_;
        bool collect_gap_samples_ = false;
        std::vector<double> gap_mid_x_samples_;
        std::vector<double> gap_mid_y_samples_;
        std::vector<double> gap_width_samples_;
        double gap_min_width_;
        double gap_max_width_;
        double gap_near_max_range_;
        double gap_search_half_angle_;
        double gap_detection_timeout_;
        double gap_sample_max_spread_;
        double gap_wall_min_length_;
        double gap_wall_max_residual_;
        double gap_wall_max_line_offset_;
        double gap_max_lateral_offset_;
        double gap_min_forward_offset_;
        double gap_max_forward_offset_;
        double lidar_offset_x_;
        double lidar_offset_y_;
        double lidar_yaw_;
        int gap_required_samples_;
        int gap_wall_min_points_;

        // 仅供 GotoD 导航巡线粗起点失败时返回最近一次 GotoC 观察点。
        bool last_parking_observation_valid_ = false;
        double last_parking_observation_x_ = 0.0;
        double last_parking_observation_y_ = 0.0;
        double last_parking_observation_yaw_ = 0.0;
        bool goto_d_recovery_pending_ = false;

        // 状态标志
        int fusion_size = 0;

        // PID与坐标相关
        Point P;
        PID Aid_X;
        PID Aid_Y;
        PID pid_center_x_;  // 用于center_x调节的PID
        const double target_center_x_ = 320.0;  // 目标中心x值（固定为320）
        double current_center_x_;  // 当前中心x值（从参数服务器读取）
};

#endif

