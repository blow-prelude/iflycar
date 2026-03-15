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
#include "ourgoal/getLaserPoint.h"


//-------------------------------------------------宏定义-----------------------------------------------
// 导航目标点

#define goto_A sendPos(&ac, 1.0, 0.5, 1.0, 0.0)
#define goto_B sendPos(&ac, 1.0, 3.0, 0.707, 0.707)
#define goto_C sendPos(&ac, 1.0, 3.0, 0.0, 1.0)
#define goto_D1 sendPos(&ac, 3.0, 4.0, 0.707, 0.707)
#define goto_D2 sendPos(&ac, 4.0, 4.0, 0.707, 0.707)
#define goto_E1 sendPos(&ac, 2.5, 3.5, -0.707, 0.707)
#define goto_E2 sendPos(&ac, 4.0, 3.5, -0.707, 0.707)
#define goto_F sendPos(&ac, 3.7, 0.0, -0.707, 0.707)

// 定义物品类别的ID。
#define CMD_Fruits 1170   // 水果
#define CMD_Vegetables 1171   // 蔬菜
#define CMD_sweet 1172 // 甜品

// 定义具体物品ID
// 水果
#define CMD_Apple 150   // 苹果
#define CMD_Banana 153   // 香蕉
#define CMD_Watermelon 155   // 西瓜

#define CMD_Apple_Apple 130   // 苹果+苹果
#define CMD_Apple_Banana 133   // 苹果+香蕉
#define CMD_Apple_Watermelon 135   // 苹果+西瓜
#define CMD_Banana_Banana 136   // 香蕉+香蕉
#define CMD_Banana_Watermelon 138   // 香蕉+西瓜
#define CMD_Watermelon_Watermelon 140   // 西瓜+西瓜

// 蔬菜 
#define CMD_pepper 260    // 辣椒
#define CMD_Tomato 263    // 西红柿
#define CMD_Potato 265    // 土豆

#define CMD_pepper_pepper 520   // 辣椒+辣椒
#define CMD_pepper_Tomato 523   // 辣椒+西红柿
#define CMD_pepper_Potato 525   // 辣椒+土豆
#define CMD_Tomato_Tomato 526   // 西红柿+西红柿
#define CMD_Tomato_Potato 528   // 西红柿+土豆
#define CMD_Potato_Potato 530   // 土豆+土豆

// 甜品 
#define CMD_Milk 370    // 牛奶
#define CMD_Cake 373    // 蛋糕 
#define CMD_coke 375    // 可乐

#define CMD_Milk_Milk 740   // 牛奶+牛奶
#define CMD_Milk_Cake 743   // 牛奶+蛋糕
#define CMD_Milk_coke 745   // 牛奶+可乐
#define CMD_Cake_Cake 746   // 蛋糕+蛋糕
#define CMD_Cake_coke 748   // 蛋糕+可乐
#define CMD_coke_coke 750   // 可乐+可乐

// 仿真
#define CMD_Gazebo1 851   // 1房间
#define CMD_Gazebo2 852   // 2房间
#define CMD_Gazebo3 853   // 3房间

//路口
#define CMD_Intersection1 951   // 路口1
#define CMD_Intersection2 952   // 路口2



// 表示任务结束的标志
#define CMD_OVER ((4 << 3) + 0)

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseAction;
// ------------------------------原服务-----------------------------------------------------
// 用于创建 ROS 服务客户端，通过这些客户端可以调用其他节点提供的服务
ros::ServiceClient play_flag_client;  // 启动语音播放。
ros::ServiceClient param_reload_client_;  // 重新加载参数

ros::ServiceClient vision_gettool_client_;  // 获取工具位置。

ros::ServiceClient teb_param_reloader;  // 重新加载TEB参数。

ros::ServiceClient getPosition_client;  // 获取当前位置



// 信息的发布
ros::Publisher cancel_pub;  // 发布取消导航目标的消息。
ros::Publisher cmd_vel_pub__;  // 发布速度命令消息。

ros::Subscriber vision_zbar_sub;

// 存储速度命令。
geometry_msgs::Twist cmd_vel;
geometry_msgs::Twist vision_vel;

// 存储位姿信息。
/*
包含了时间戳（stamp）和参考坐标系（frame_id）等信息。
包含了位置（position）、姿态（orientation）以及协方差矩阵（covariance）。

pose_msg：这个变量一般用来存储经过处理或者校正后的位姿信息。
例如，在机器人定位中，可能会对原始的位姿数据进行滤波、校正等操作，之后将处理好的结果存于pose_msg里。

raw_pose_msg：该变量通常用于存储未经处理的原始位姿信息。比如，传感器直接测量得到的位姿数据，就可以先存于raw_pose_msg中。
*/
geometry_msgs::PoseWithCovarianceStamped pose_msg;
geometry_msgs::PoseWithCovarianceStamped raw_pose_msg;


// ---------------------------------原服务通信数据------------------------------------


// 存储从服务调用中获取的数据。
ourgoal::getPosition position_srv;       //  标志牌的位姿
ourgoal::srv_reload teb_reload;

ourgoal::getLaserPoint vision_srv;

std_srvs::Empty _;

//  -----------------------------------------我的服务通信数据------------------------------------



// 创建一个空的GoalID消息
/*
用于标识动作目标的消息类型。
cancel_msg 变量用于存储取消动作目标的消息。
*/
actionlib_msgs::GoalID cancel_msg;
/*
getToolTime 变量用于记录获取工具操作开始的时间。在后续的代码中，可能会使用这个时间来判断获取工具的操作是否超时。
*/
ros::Time getToolTime;


/*
tool：初始值为 0，用于存储工具的 ID。
target：用于存储物品ID。
*/
int tool = 0, target;
int target_class;  // 记录本次任务的物品类别
int target1;     // 记录现实中取到的物品
int target2;     // 记录仿真中取到的物品
int target3;     // 最终的结果

double roll, pitch, yaw;
/*
vision_gettool  是否 在拣货区路口  通过视觉成功获取到工具。

*/
int vision_gettool = 0;
int vision_getIntersection1 = 0;
int vision_getIntersection2 = 0;
/*
targetX 和 targetY 分别表示目标点的 X 坐标和 Y 坐标。
*/
double targetX, targetY;
int fusion_size = 0;
int Point_count = 0;

typedef struct Point
{
    double px;
    double py;
    double oz;
    double ow;
} Point;

Point P;

typedef struct
{
    double kp, ki, kd;   // 三个系数
    float err, err_last; // 误差、上次误差
    float integral;      // 积分
    float value;         // 输出
} PID;

PID Aid_X;      // 将小车精确地停在与目标一定距离的地方
PID Aid_Y;


//-----------------------------------------------------其他函数-------------------------------------------------------

// 暂停程序执行指定的秒数。
/*
输入：暂停的秒数
*/
void delayedFunction(int delayInSeconds)
{
    std::this_thread::sleep_for(std::chrono::seconds(delayInSeconds));
    // 执行延迟后的操作
}

// 打印数组
void PrintfArray(float arr[], int size)
{
    printf("size:%d\n", size);
    for (int i = 0; i < size; i++)
    {
        printf("%f\n", arr[i]);
    }
}

// 限制输出值
double Limit_Value(double INPUT, double MAX, double MIN)
{
    if (INPUT > MAX)
    {
        INPUT = MAX;
    }
    else if (INPUT < MIN)
    {
        INPUT = MIN;
    }
    return INPUT;
}

double PID_Realize(PID *pid, double err, double MAX, double MIN)
{
    // 设置目标角速度，计算误差，更新PID输出
    pid->err = err;                                                                                     // 误差计算
    pid->integral += pid->err;                                                                          // 积分
    double value = pid->kp * pid->err + pid->ki * pid->integral + pid->kd * (pid->err - pid->err_last); // PID实现
    pid->err_last = pid->err;                                                                           // 误差更新
    pid->value = value;
    return Limit_Value(pid->value, MAX, MIN);
}


/*
接收经过点云处理后，周围目标点的全局坐标
*/
void getPoint(Point *P)
{
    ros::NodeHandle nh;

    getPosition_client.call(position_srv);

    fusion_size = position_srv.response.fusion_size;  // 融合后的目标点数量

    if (fusion_size != 0)
    {
        P->px = position_srv.response.px;
        P->py = position_srv.response.py;
        P->ow = position_srv.response.ow;
        P->oz = position_srv.response.oz;
    }
}

/*
表示坐标点
*/
typedef struct point_2d
{
    double x;
    double y;
} point_2d;


// 坐标平移函数
point_2d translate(point_2d p, double dx, double dy)
{
    p.x += dx;
    p.y += dy;
    return p;
}

// 坐标旋转函数
// 将输入的二维点 p 绕原点旋转 CarYaw 角度。
point_2d rotate(point_2d p, double CarYaw)
{
    double rad = 1 * CarYaw;
    double NewPointX, NewPointY;
    NewPointX = p.x * cos(rad) - p.y * sin(rad);
    NewPointY = p.x * sin(rad) + p.y * cos(rad);

    p.x = NewPointX;
    p.y = NewPointY;
    return p;
}

// 四元数结构体定义
struct Quaternion
{
    double w, x, y, z;
};

// 欧拉角转四元数函数
Quaternion eulerToQuaternion(double roll, double pitch, double yaw)
{
    // 计算中间值
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);

    Quaternion q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

    return q;
}

//------------------------------------------------------Action-------------------------------------------------------------
/*
以map为参考系  
发送导航点  move_base_msgs::MoveBaseGoal
x, y：目标点的二维坐标（x 和 y）
z, w：目标点的方向（四元数的 z 和 w 分量）
*/
void sendPos(actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> *ac, double x, double y, double z, double w)
{
    move_base_msgs::MoveBaseGoal goal;
    goal.target_pose.header.stamp = ros::Time::now();
    goal.target_pose.header.frame_id = "map";

    goal.target_pose.pose.position.x = x;
    goal.target_pose.pose.position.y = y;
    goal.target_pose.pose.position.z = 0;
    goal.target_pose.pose.orientation.x = 0;
    goal.target_pose.pose.orientation.y = 0;
    goal.target_pose.pose.orientation.z = z;
    goal.target_pose.pose.orientation.w = w;

    ROS_INFO("Sending goal\n");
    ac->sendGoal(goal);
}

/*
默认情况下，第一个常量的值为 0
记录小车当前的任务阶段
*/
enum State
{
    TEST_,      // 平时用来测试，比赛应该要删掉
    GOTOA_,    // 到达领取采购任务区  获取ID  语音播报
    GOTOB_,    // 到达拣货区路口  停下（取消导航点发布） 
    VISION_GETTOOL_,   // 说明机器人通过视觉系统来获取工具的状态，会利用摄像头等视觉传感器进行标志牌的识别和定位。
    GOTOC_,    //  到达仿真区进行仿真
    Gazebo_,   //  仿真    语音播报
    GOTOD_,    // 到达1号标牌前   查看红绿灯   语音播报
    VISION_LINE_,    // 表示机器人进行视觉巡线的状态，
    GOTOF_,     // 到达终点   路中间会随机出现障碍物，要有避障算法    进行价格计算  语音播报
};
State current_state;

void InitPID()
{
    ros::NodeHandle nh;
    Aid_X.kp = 0;
    Aid_X.ki = 0;
    Aid_X.kd = 0;
    Aid_X.err = 0;
    Aid_X.err_last = 0;
    Aid_X.integral = 0;
    Aid_X.value = 0;

    Aid_Y.kp = 0;
    Aid_Y.ki = 0;
    Aid_Y.kd = 0;
    Aid_Y.err = 0;
    Aid_Y.err_last = 0;
    Aid_Y.integral = 0;
    Aid_Y.value = 0;


    nh.param("Aid_X_kp", Aid_X.kp, 1.0);
    nh.param("Aid_X_ki", Aid_X.ki, 0.0);
    nh.param("Aid_X_kd", Aid_X.kd, 0.0);
    nh.param("Aid_Y_kp", Aid_Y.kp, 1.0);
    nh.param("Aid_Y_ki", Aid_Y.ki, 0.0);
    nh.param("Aid_Y_kd", Aid_Y.kd, 0.0);
}

/*
把 到达坐标点   ID获取  语音播报进行封装
*/
void GotoA()
{
    ros::NodeHandle nh;
    MoveBaseAction ac(nh, "move_base", true);

    printf("gotoA!!!\n");

    // teb_reload.request.ask = 0;
    // teb_param_reloader.call(teb_reload);
    // ROS_WARN("teb_param0_reload!!!");

    goto_A;
    ac.waitForResult();
    while (!(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
        ;

    
    do
    {
        nh.getParam("target_class", target_class); // 得到物品ID
    } while (target_class != CMD_Fruits && target_class != CMD_Vegetables && target_class != CMD_sweet);


    ROS_WARN("Targetis %d!", target_class);

    
    // teb_reload.request.ask = 1;
    // teb_param_reloader.call(teb_reload);
    // ROS_WARN("teb_param1_reload!!!");


    nh.setParam("target_tool", target_class);     // 语音播报会接受target_tool 来进行语音播报

    nh.setParam("audio", 0); // 语音运行中
    play_flag_client.call(_);
    while (!nh.param("audio", 0))
        ;
}


/*
封装   到达目标点   取消当行目标让小车停下   让视觉去看ID
*/
void GotoB()
{
    ros::NodeHandle nh;
    MoveBaseAction ac(nh, "move_base", true);
    printf("gotoAid\n");

    goto_B;
    while (!(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
        ;

    while (!vision_gettool)  //  一直循环，知道视觉找到目标
    {   
        // 如果没有找到目标，就一直旋转，直到找到目标
        do
        {
            cancel_pub.publish(cancel_msg);   // 这一句可能不需要加
            cmd_vel.linear.x = 0;
            cmd_vel.linear.y = 0;
            cmd_vel.linear.z = 0;
            cmd_vel.angular.x = 0;
            cmd_vel.angular.y = 0;
            cmd_vel.angular.z = 0.5;
            cmd_vel_pub__.publish(cmd_vel);
        } while (nh.param("/vision_gettool", 0)==0);

        if (nh.param("/vision_gettool", 0))   //  视觉找到目标
        {
            vision_gettool = 1;
            printf("vision_gettool:%d\n", vision_gettool);

            // 发布消息以取消导航目标
            cancel_pub.publish(cancel_msg);
            current_state = VISION_GETTOOL_;

        }
    }

}


// 接收视觉信息，在标志牌前停稳
void Vision_GetTool()
{
    printf("vision_gettool\n");
    ros::NodeHandle nh;
    MoveBaseAction ac(nh, "move_base", true);

    delayedFunction(3); // 延迟3秒钟   可能是为了等待某些操作完成或者系统稳定。

    // 从 ROS 参数服务器中获取机器人当前的偏航角 CarYaw、x 坐标 CarX、y 坐标 CarY、z 坐标 CarZ 和四元数的 w 分量 CarW，
    // 这些参数用于后续的坐标变换和姿态计算。
    double CarYaw = nh.param("CarYaw", 0.0);
    double CarX = nh.param("CarX", 0.0);
    double CarY = nh.param("CarY", 0.0);
    double CarZ = nh.param("CarZ", 0.0);
    double CarW = nh.param("CarW", 0.0);


    // 定义了三个 point_2d 类型的变量 target_point、point_1 和 point_2，用于存储坐标点。
    point_2d target_point, point_1, point_2;

    // 计算目标点 target_point 的 x、y 坐标，取 point_1 和 point_2 坐标的平均值。
    // 激光雷达坐标系下的坐标
    point_1.x = nh.param("dx1", 0.0);
    point_2.x = nh.param("dx2", 0.0);
    point_1.y = nh.param("dy1", 0.0);
    point_2.y = nh.param("dy2", 0.0);

    target_point.x = (point_2.x + point_1.x) / 2;
    target_point.y = (point_2.y + point_1.y) / 2;


    // 下面应该是根据视觉让小车斜着停的策略   但我每看懂
    // 这个参数是通过视觉系统得到的一个与工具角度相关的值。

    // 从参数服务器获取视觉检测线斜率
    double k = nh.param("vision_tool_a", 0);
    double kk = 0;
    // 从参数服务器获取视觉检测线斜率
    if (k == 255)
        kk = M_PI / 2;
    else
        kk = std::atan(k);
    // 角度规范化（0~π）
    if (kk < 0)
        kk += M_PI;
    // 转换为车辆坐标系角度（-π/2~π/2）
    kk -= M_PI / 2;

    // 保持30cm安全距离
    double distance = 0.30;
    target_point.x -= distance * std::cos(kk);
    target_point.y -= distance * std::sin(kk);

    printf("targetX/Y:%lf/%lf\n", target_point.x, target_point.y);
    printf("CarYaw:%lf\n", CarYaw / M_PI * 180);

    printf("k:%lf\n", k / M_PI * 180);
    printf("kk:%lf\n", kk / M_PI * 180);
    printf("!!!targetX/Y:%lf/%lf\n", target_point.x, target_point.y);

    // 转化为全局坐标
    target_point = rotate(target_point, CarYaw);
    target_point = translate(target_point, CarX, CarY);

    // 合成最终航向角
    double vision_yaw = kk + CarYaw;  // 目标方向 = 车辆航向 + 相对角度


    struct Quaternion q = eulerToQuaternion(0, 0, vision_yaw);

    sendPos(&ac, target_point.x, target_point.y, q.z, q.w);
    printf("Target_xyzw:%lf,%lf,%lf,%lf\n", target_point.x, target_point.y, q.z, q.w);
    while (!(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
        ;


    do
    {
        nh.getParam("target", target); // 得到物品ID
    } while (target != CMD_Apple && target != CMD_Banana && target != CMD_Watermelon && target != CMD_pepper && target != CMD_Tomato && target != CMD_Potato && target != CMD_Milk && target != CMD_Cake && target != CMD_coke);


    ROS_WARN("Targetis %d!", target);

    target1 = target; // 记录现实中取到的物品

    nh.setParam("target_tool", target);

    nh.setParam("audio", 0); // 语音运行中
    play_flag_client.call(_);
    while (!nh.param("audio", 0))
        ;

}

void GotoC()
{
    ros::NodeHandle nh;
    MoveBaseAction ac(nh, "move_base", true);
    

    

    goto_C;
    ac.waitForResult();
    while (!(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
        ;

    // 速度环PID微调     让小车能精准停经仿真框里
    // cancel_pub.publish(cancel_msg);
    // double qian, you;
    // do
    // {
    //     qian = nh.param("ErrorX", 0.0) - 1.00;
    //     you = nh.param("ErrorY", 0.0) - 2;
    //     cmd_vel.linear.x = PID_Realize(&Aid_X, qian, 0.8, -0.8);
    //     cmd_vel.linear.y = -PID_Realize(&Aid_Y, you, 0.8, -0.8);
    //     cmd_vel.linear.z = 0;
    //     cmd_vel.angular.x = 0;
    //     cmd_vel.angular.y = 0;
    //     cmd_vel.angular.z = 0;
    //     cmd_vel_pub__.publish(cmd_vel);
    // } while (std::fabs(qian) > 0.05 && std::fabs(you) > 0.05);
   
}

void Gazebo()
{
   //    .......(进入仿真)
   ros::NodeHandle nh;


   
   do
   {
       nh.getParam("target", target); // 得到物品ID
   } while (target != CMD_Gazebo1 && target != CMD_Gazebo2 && target != CMD_Gazebo3);

   target2 = target; // 记录仿真中取到的物品
   nh.setParam("target_tool", target);

   nh.setParam("audio", 0); // 语音运行中
   play_flag_client.call(_);
    while (!nh.param("audio", 0))
        ;
}

void GotoD()
{
    ros::NodeHandle nh;
    MoveBaseAction ac(nh, "move_base", true);
    printf("gotoD\n");

    goto_D1;
    while (!(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
        ;

    //第一个红绿灯前
    while (!(vision_getIntersection1||vision_getIntersection2))  //  一直循环，知道视觉找到目标
    {   

        if (nh.param("/vision_getIntersection1", 0))   //  视觉找到目标
        {
            vision_getIntersection1 = 1;
            target = CMD_Intersection1;
            printf("vision_getIntersection1:%d\n", vision_getIntersection1);

        }
        if (nh.param("/vision_getIntersection2", 0))   //  视觉找到目标
        {
            vision_getIntersection2 = 1;
            target = CMD_Intersection2;
            printf("vision_getIntersection2:%d\n", vision_getIntersection2);

        }
    }

    if (vision_getIntersection1==1)   // 走路口1说明是绿灯
    {   
        vision_getIntersection1 = 0;  // 复位
        printf("green\n");

        nh.setParam("target_tool", target);     // 语音播报会接受target_tool 来进行语音播报

        nh.setParam("audio", 0); // 语音运行中
        play_flag_client.call(_);
        while (!nh.param("audio", 0))
        ;

        goto_E1; // 巡线起点
        while (!(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
        ;
    }
    if(vision_getIntersection2==1)   //  红灯
    {
        vision_getIntersection2 = 0;
        printf("red\n");
        goto_D2; // 巡线起点
         while (!(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))

        nh.setParam("target_tool", target);    
        nh.setParam("audio", 0); // 语音运行中
        play_flag_client.call(_);
        while (!nh.param("audio", 0))
        ;
        goto_E2; // 巡线起点
        while (!(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
       ;
    }
   
}


void vision_line()
{
    // teb_reload.request.ask = 0;       // 重新加载TEB参数
    // teb_param_reloader.call(teb_reload);

    ros::Duration(1.0).sleep();

    ros::param::set("vision_line", 1);
    // system("rosnode kill getyolo");

    // system("roslaunch vision_line vision_line.launch");
    // printf("vision!!!\n");
}

void GotoF()
{
    ros::NodeHandle nh;
    MoveBaseAction ac(nh, "move_base", true);
    printf("gotoAid\n");


    teb_reload.request.ask = 1;       // 重新加载TEB参数  速度降下来避障
    teb_param_reloader.call(teb_reload);

    goto_F;
    while (!(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
        ;

    target3 = target1 + target2; // 记录现实中取到的物品+仿真中取到的物品
    nh.setParam("target_tool", target3);     // 语音播报会接受target_tool 来进行语音播报

    nh.setParam("audio", 0); // 语音运行中
    play_flag_client.call(_);
    while (!nh.param("audio", 0))
        ;
}


// 二维码回调函数
void getZbarCallback(const std_msgs::String::ConstPtr &msg) //给sub的回调函数，名字自己定义，参数是订阅到的消息
{
    ros::NodeHandle nh;
    // 将接收到的消息转换为字符串
    std::string received_msg = msg->data;
    
    // 根据接收到的消息设置相应的标志位
    if (received_msg == "Fruit") {
        nh.setParam("target_class", CMD_Fruits);
        ROS_INFO("Fruit");
       
    } 
    else if (received_msg == "Vegetable") {
        nh.setParam("target_class", CMD_Vegetables);
        ROS_INFO("Vegetable");
        
    } 
    else if (received_msg == "Dessert") {
        nh.setParam("target_class", CMD_sweet);
        
    } 
    else {
        ROS_WARN("Received unknown message: ");
    }
 

}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "switch_node");
    ros::NodeHandle nh;

    ros::AsyncSpinner spinner(1);     
    spinner.start();

    current_state = GOTOA_;

    position_srv.request.ask = true;

    // clear_costmaps_client = nh.serviceClient<std_srvs::Empty>("/move_base/clear_costmaps");


    nh.setParam("yolo_begin", 0);
    vision_gettool_client_ = nh.serviceClient<ourgoal::getLaserPoint>("/srv_getLaserPoint");

    // 发布消息
    cancel_pub = nh.advertise<actionlib_msgs::GoalID>("move_base/cancel", 10);
    cmd_vel_pub__ = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);


    // 订阅消息
    teb_param_reloader = nh.serviceClient<ourgoal::srv_reload>("/param_reload");
    getPosition_client = nh.serviceClient<ourgoal::getPosition>("/srv_getPosition");   // 获取目标牌位姿

    // 接收话题
    vision_zbar_sub = nh.subscribe("/vision_zbar", 10, getZbarCallback);

    ros::service::waitForService("/param_reload");
    // teb_reload.request.ask = 0;
    // teb_param_reloader.call(teb_reload);
    InitPID();
    ROS_WARN("init ok!!!");

    MoveBaseAction ac(nh, "move_base", true);
    // 语音启动
    play_flag_client = nh.serviceClient<std_srvs::Empty>("/play_flag_srv");
    ac.waitForServer(ros::Duration(5)); // 语音唤醒延迟

    while (!nh.param("awake", 0))
        ;
    ROS_WARN("RUN!");

    ros::Rate loop_rate(10);

    while (ros::ok())
    {
        switch (current_state)
        {
        case TEST_:
        // while (!nh.param("awake", 0))
        //     ;
        break;
        case GOTOA_:
            GotoA();  
            current_state = GOTOB_;
            break;
        case GOTOB_:
            GotoB();   // 函数里面写了状态转换
            break;
        case VISION_GETTOOL_:
            Vision_GetTool();
            current_state = GOTOC_;
            break;
        case GOTOC_:
            GotoC();
            current_state = Gazebo_;
            break;
        case Gazebo_:
            Gazebo();
           current_state = GOTOD_;   
           break;
        case GOTOD_:
            GotoD();
            current_state = VISION_LINE_;
            break;
        case VISION_LINE_:
            vision_line();
            current_state = GOTOF_;
            break;
        case GOTOF_:
            GotoF();
            current_state = TEST_;
            break;
        }

        loop_rate.sleep();
    }
    return 0;
}
