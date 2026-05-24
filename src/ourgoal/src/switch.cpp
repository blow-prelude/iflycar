#include "switch.h"

// 导航点宏定义
#define goto_A1 sendPos(1.7, -0.05, -0.0872, 0.9962)
#define goto_A sendPos(0.70, 0.47, 1.0, 0.0)
#define goto_B sendPos(0.86, 3.25, 0.707, 0.707)
#define goto_C sendPos(0.96, 3.50, 0.0, 1.0)
#define goto_D1 sendPos(3.15, 4.0, 0.707, 0.707)
#define goto_D2 sendPos(4.2, 4.0, 0.707, 0.707)
#define goto_E1 sendPos(3.0, 3.2, -0.7071, 0.7071)
#define goto_E2 sendPos(4.4, 3.2, -0.7071, 0.7071) 
#define goto_F sendPos(3.50, -0.90, -0.707, 0.707)

// 物品ID宏定义
#define CMD_Fruits 1170
#define CMD_Vegetables 1171
#define CMD_sweet 1172
#define CMD_Apple 530
#define CMD_Banana 609
#define CMD_Watermelon 1086
#define CMD_pepper 652
#define CMD_Tomato 660
#define CMD_Potato 663
#define CMD_Milk 429
#define CMD_Cake 404
#define CMD_coke 418
#define CMD_Gazebo1 1
#define CMD_Gazebo2 2
#define CMD_Gazebo3 3
#define CMD_Intersection1 951  // 对应绿色信号灯
#define CMD_Intersection2 952  // 对应红色信号灯
#define CMD_OVER ((4 << 3) + 0)

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseAction;

// 构造函数
OURSWITCH::OURSWITCH()
    : ac_(nh_, "move_base", true)
{
    ROS_INFO("switch node initialized");

    current_state = GOTOA_;
    nh_.setParam("yolo_begin", 0);
    position_srv.request.ask = true;
    current_color_ = 0;  // 初始化颜色结果（0=未识别）

    // 初始化视觉服务客户端
    vision_gettool_client_ = nh_.serviceClient<ourgoal::getLaserPoint>("/srv_getLaserPoint");

    // 初始化center_x控制的PID参数
    pid_center_x_.kp = 0.0005;   // 比例系数
    pid_center_x_.ki = 0.0001;  // 积分系数
    pid_center_x_.kd = 0.001;   // 微分系数
    pid_center_x_.err = 0;
    pid_center_x_.err_last = 0;
    pid_center_x_.integral = 0;
    pid_center_x_.output = 0;

    // 初始化距离控制的PID参数
    pid_distance_.kp = 0.25;     // 距离控制比例系数
    pid_distance_.ki = 0.02;    // 距离控制积分系数
    pid_distance_.kd = 0.01;     // 距离控制微分系数
    pid_distance_.err = 0;
    pid_distance_.err_last = 0;
    pid_distance_.integral = 0;
    pid_distance_.output = 0;

    // 初始化发布者
    cancel_pub = nh_.advertise<actionlib_msgs::GoalID>("move_base/cancel", 10);
    cmd_vel_pub__ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
    Gazebo_Command = nh_.advertise<std_msgs::Int32>("class_command", 10);

    // 初始化订阅者
    teb_param_reloader = nh_.serviceClient<ourgoal::srv_reload>("/param_reload");
    getPosition_client = nh_.serviceClient<ourgoal::getPosition>("/srv_getPosition");
    vision_zbar_sub = nh_.subscribe("/vision_zbar", 10, &OURSWITCH::getZbarCallback, this);
    Gazebo_Room = nh_.subscribe("/received_room_num", 10, &OURSWITCH::getGzeboRoomCallback, this);
    Gazebo_Food = nh_.subscribe("/changed_food_num", 10, &OURSWITCH::getGzeboFoodCallback, this);
    vision_angle = nh_.subscribe("/vision_angle", 1024, &OURSWITCH::visionAngleCallback, this);
    sub_ultrasound = nh_.subscribe("/ultra", 10, &OURSWITCH::UltrasoundCallback, this);
    // 新增：订阅颜色识别结果话题
    color_result_sub = nh_.subscribe("/color_result", 10, &OURSWITCH::colorResultCallback, this);

    Start_CarX = nh_.param("CarX", 0.0);
    Start_CarY = nh_.param("CarY", 0.0);

    ros::service::waitForService("/param_reload");
    ROS_WARN("Initialization complete!");

    // 语音启动
    play_flag_client = nh_.serviceClient<std_srvs::Empty>("/play_flag_srv");
    ac_.waitForServer(ros::Duration(5));
    while (!nh_.param("awake", 0));
    ROS_WARN("System started!");
}

// 析构函数
OURSWITCH::~OURSWITCH()
{
    ROS_WARN("switch node terminated");
}

// 延迟函数
void OURSWITCH::delayedFunction(int delayInSeconds)
{
    std::this_thread::sleep_for(std::chrono::seconds(delayInSeconds));
}

// 打印数组
void OURSWITCH::PrintfArray(float arr[], int size)
{
    printf("Array size: %d\n", size);
    for (int i = 0; i < size; i++)
    {
        printf("Element %d: %f\n", i, arr[i]);
    }
}

// 限制输出值
double OURSWITCH::Limit_Value(double INPUT, double MAX, double MIN)
{
    if (INPUT > MAX) return MAX;
    else if (INPUT < MIN) return MIN;
    return INPUT;
}

// PID实现（基础版）
double OURSWITCH::PID_Realize(PID *pid, double err, double MAX, double MIN)
{
    pid->err = err;
    pid->integral += pid->err;
    double output = pid->kp * pid->err + pid->ki * pid->integral + pid->kd * (pid->err - pid->err_last);
    pid->err_last = pid->err;
    pid->output = output;
    return Limit_Value(pid->output, MAX, MIN);
}

// PID实现（带积分限幅）
double OURSWITCH::PID_Realize2(PID *pid, double err, double MAX, double MIN, double integral_limit)
{
    pid->err = err;
    pid->integral += pid->err;
    
    // 积分限幅
    if (fabs(pid->integral) > integral_limit)
        pid->integral = (pid->integral > 0) ? integral_limit : -integral_limit;
        
    double output = pid->kp * pid->err + pid->ki * pid->integral + pid->kd * (pid->err - pid->err_last);
    pid->err_last = pid->err;
    pid->output = output;
    return Limit_Value(pid->output, MAX, MIN);
}

// PID初始化
void OURSWITCH::InitPID()
{
    // 初始化Aid_X的PID参数
    Aid_X.kp = 0;
    Aid_X.ki = 0;
    Aid_X.kd = 0;
    Aid_X.err = 0;
    Aid_X.err_last = 0;
    Aid_X.integral = 0;
    Aid_X.output = 0;

    // 初始化Aid_Y的PID参数
    Aid_Y.kp = 0;
    Aid_Y.ki = 0;
    Aid_Y.kd = 0;
    Aid_Y.err = 0;
    Aid_Y.err_last = 0;
    Aid_Y.integral = 0;
    Aid_Y.output = 0;

    // 从参数服务器加载PID参数
    nh_.param("Aid_X_kp", Aid_X.kp, 1.0);
    nh_.param("Aid_X_ki", Aid_X.ki, 0.0);
    nh_.param("Aid_X_kd", Aid_X.kd, 0.0);
    nh_.param("Aid_Y_kp", Aid_Y.kp, 1.0);
    nh_.param("Aid_Y_ki", Aid_Y.ki, 0.0);
    nh_.param("Aid_Y_kd", Aid_Y.kd, 0.0);
}

// 从参数服务器获取center_x
bool OURSWITCH::getCenterXFromParam()
{
    // 尝试从参数服务器获取center_x
    if (nh_.getParam("center_x", current_center_x_))
    {
        // 检查值是否有效（假设图像宽度为640像素）
        if (current_center_x_ < 0 || current_center_x_ > 640)
        {
            ROS_WARN("Invalid center_x value: %.2f (out of 0-640 range)", current_center_x_);
            return false;
        }
        return true;
    }
    else
    {
        ROS_WARN("Failed to get center_x from parameter server");
        return false;
    }
}

// 计算y方向速度（用于center_x调节）
double OURSWITCH::calculateYVelocity()
{
    // 计算误差
    double error = target_center_x_ - current_center_x_;
    
    // 积分项（带限幅）
    pid_center_x_.integral += error;
    const double integral_limit = 200.0; // 积分限幅
    if (pid_center_x_.integral > integral_limit)
        pid_center_x_.integral = integral_limit;
    else if (pid_center_x_.integral < -integral_limit)
        pid_center_x_.integral = -integral_limit;
    
    // 微分项
    double derivative = error - pid_center_x_.err_last;
    
    // PID输出计算
    double output = pid_center_x_.kp * error + 
                   pid_center_x_.ki * pid_center_x_.integral + 
                   pid_center_x_.kd * derivative;
    
    // 保存当前误差
    pid_center_x_.err_last = error;
    
    // 速度限幅
    const double max_y_vel = 0.1; // 最大y方向速度
    return Limit_Value(output, max_y_vel, -max_y_vel);
}

// 计算x方向速度（用于距离调节）
double OURSWITCH::calculateXVelocity()
{
    // 计算距离误差（目标距离 - 当前距离）
    double error = target_distance_ - distance_qian_x;
    
    // 积分项（带限幅）
    pid_distance_.integral += error;
    const double integral_limit = 1.0; // 积分限幅
    if (pid_distance_.integral > integral_limit)
        pid_distance_.integral = integral_limit;
    else if (pid_distance_.integral < -integral_limit)
        pid_distance_.integral = -integral_limit;
    
    // 微分项
    double derivative = error - pid_distance_.err_last;
    
    // PID输出计算
    double output = pid_distance_.kp * error + 
                   pid_distance_.ki * pid_distance_.integral + 
                   pid_distance_.kd * derivative;
    
    // 保存当前误差
    pid_distance_.err_last = error;
    
    // 速度限幅（限制在安全范围内）
    const double max_x_vel = 0.1; // 最大x方向速度
    return Limit_Value(output, max_x_vel, -max_x_vel);
}

// 获取点云处理后的目标点
void OURSWITCH::getPoint(Point *P)
{
    getPosition_client.call(position_srv);
    fusion_size = position_srv.response.fusion_size;
    
    if (fusion_size != 0)
    {
        P->px = position_srv.response.px;
        P->py = position_srv.response.py;
        P->ow = position_srv.response.ow;
        P->oz = position_srv.response.oz;
    }
}

// 坐标平移
point_2d OURSWITCH::translate(point_2d p, double dx, double dy)
{
    p.x += dx;
    p.y += dy;
    return p;
}

// 坐标旋转
point_2d OURSWITCH::rotate(point_2d p, double CarYaw)
{
    double rad = CarYaw;
    double NewPointX = p.x * cos(rad) - p.y * sin(rad);
    double NewPointY = p.x * sin(rad) + p.y * cos(rad);
    p.x = NewPointX;
    p.y = NewPointY;
    return p;
}

// 欧拉角转四元数
Quaternion OURSWITCH::eulerToQuaternion(double roll, double pitch, double yaw)
{
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

// 发送导航目标点
void OURSWITCH::sendPos(double x, double y, double z, double w)
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

    ROS_INFO("Sending goal: x=%.2f, y=%.2f, z=%.2f, w=%.2f", x, y, z, w);
    ac_.sendGoal(goal);
}

// 到达领取采购任务区
void OURSWITCH::GotoA()
{
    ROS_INFO("Entering GotoA state");
    
    // teb_reload.request.ask = 1;
    // teb_param_reloader.call(teb_reload);
    // ROS_WARN("teb_param1_reload!!!");

    // 让车向前走1秒
    ROS_INFO("Moving forward for 2 second");
    ros::Time rotate_start = ros::Time::now();
    while (ros::Time::now() - rotate_start < ros::Duration(3.5))
    {
	cancel_pub.publish(cancel_msg);
	cmd_vel.linear.x = 0.4;
	cmd_vel.linear.y = 0.0;
	cmd_vel.angular.z = 0.0;  
	cmd_vel_pub__.publish(cmd_vel);
        ros::spinOnce();
    }
    
    // 停止移动
    cmd_vel.linear.x = 0.0;
    cmd_vel_pub__.publish(cmd_vel);
    
    // 继续原有的导航逻辑
    // goto_A1;
    // ac_.waitForResult();
    // while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
    //    ros::spinOnce();
    
    goto_A;
    ac_.waitForResult();
    while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
        ros::spinOnce();

    // 获取物品类别
    do
    {
        nh_.getParam("target_class", target_class);
    } while (target_class != CMD_Fruits && target_class != CMD_Vegetables && target_class != CMD_sweet);

    ROS_WARN("Target class identified: %d", target_class);
    nh_.setParam("target_tool", target_class);
    
    // 语音播报
    nh_.setParam("audio", 0);
    play_flag_client.call(_);
    while (!nh_.param("audio", 0))
        ros::spinOnce();
}


// 到达拣货区路口
void OURSWITCH::GotoB()
{
    ROS_INFO("Entering GotoB state");

    // 原来的参数
    // teb_reload.request.ask = 0;
    // teb_param_reloader.call(teb_reload);
    // ROS_WARN("teb_param0_reload!!!");

    // 导航到B点
    // goto_B1;
    // while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
    //     ros::spinOnce();
    
    // goto_B2;
    // while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
    //     ros::spinOnce();
    
    goto_B;
    while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
        ros::spinOnce();

    nh_.setParam("yolo_begin", 1);
    vision_gettool = false;
    const double rotate_duration = 1.0;
    const double check_duration = 1.0;
    
    // 循环直到找到目标
    while (!vision_gettool && ros::ok())
    {   
        // 旋转寻找目标
        ros::Time rotate_start = ros::Time::now();
        while (ros::Time::now() - rotate_start < ros::Duration(rotate_duration))
        {
            cancel_pub.publish(cancel_msg);
            cmd_vel.linear.x = 0;
            cmd_vel.angular.z = -0.5;  // 逆时针旋转
            cmd_vel_pub__.publish(cmd_vel);
            ros::spinOnce();
        }
        
        // 停止旋转
        cmd_vel.angular.z = 0;
        cmd_vel_pub__.publish(cmd_vel);
        
        // 检查是否找到目标
        ros::Time check_start = ros::Time::now();
        while (ros::Time::now() - check_start < ros::Duration(check_duration))
        {
            if (nh_.param("/vision_gettool", 0) != 0)
            {
                vision_gettool = true;
                break;
            }
            ros::spinOnce();
        }
        
        if (vision_gettool)
        {
            ROS_INFO("Target found, switching to VISION_GETTOOL_ state");
            cancel_pub.publish(cancel_msg);
            current_state = VISION_GETTOOL_;
        }
        else
        {
            ROS_INFO("Target not found, continuing search");
        }
    }
}

// 视觉获取工具：先调整物品到中心，再调整距离
void OURSWITCH::Vision_GetTool()
{
    ROS_INFO("Entering Vision_GetTool state");
    delayedFunction(3);

    // 获取机器人当前位姿
    double CarYaw = nh_.param("CarYaw", 0.0);
    double CarX = nh_.param("CarX", 0.0);
    double CarY = nh_.param("CarY", 0.0);

    // 坐标点处理
    point_2d target_point, point_1, point_2;
    point_1.x = nh_.param("dx1", 0.0);
    point_2.x = nh_.param("dx2", 0.0);
    point_1.y = nh_.param("dy1", 0.0);
    point_2.y = nh_.param("dy2", 0.0);

    target_point.x = (point_2.x + point_1.x) / 2;
    target_point.y = (point_2.y + point_1.y) / 2;

    // 视觉角度计算
    double k = nh_.param("vision_tool_a", 0.0);
    double kk = (k == 255) ? M_PI / 2 : std::atan(k);
    if (kk < 0) kk += M_PI;
    kk -= M_PI / 2;

    // 保持30cm安全距离
    double distance = 0.30;
    target_point.x -= distance * std::cos(kk);
    target_point.y -= distance * std::sin(kk);

    // 转换为全局坐标
    target_point = rotate(target_point, CarYaw);
    target_point = translate(target_point, CarX, CarY);

    // 计算航向角并发送目标
    double vision_yaw = kk + CarYaw;
    Quaternion q = eulerToQuaternion(0, 0, vision_yaw);
    sendPos(target_point.x, target_point.y, q.z, q.w);
    
    while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
        ros::spinOnce();

    ros::Time rotate_start = ros::Time::now();
    while (ros::Time::now() - rotate_start < ros::Duration(0.5))
    {
	cancel_pub.publish(cancel_msg);
	cmd_vel.linear.x = -0.2;
	cmd_vel.linear.y = 0.0;
	cmd_vel.angular.z = 0.0;  
	cmd_vel_pub__.publish(cmd_vel);
        ros::spinOnce();
    }
    
    // 停止移动
    cmd_vel.linear.x = 0.0;
    cmd_vel_pub__.publish(cmd_vel);

    // 第一阶段：调节center_x到320（先将物品移动到像素中间）
    const double center_tolerance = 80.0; // 允许的误差范围（像素）
    const int max_center_adjust_count = 5000; // 最大调节次数
    int center_adjust_count = 0;
    bool center_adjustment_complete = false;

    ROS_INFO("=== Starting center_x adjustment to 320 ===");
    
    while (center_adjust_count < max_center_adjust_count && !center_adjustment_complete && ros::ok())
    {
        // 从参数服务器获取center_x
        if (!getCenterXFromParam())
        {
            ROS_WARN("Failed to get valid center_x, retrying...");
            center_adjust_count++;
            continue;
        }

        // 计算误差
        double error = fabs(target_center_x_ - current_center_x_);
        ROS_INFO("Center Adjustment %d: Current=%.2f, Target=%.2f, Error=%.2f",
                 center_adjust_count, current_center_x_, target_center_x_, error);

        // 检查是否达到目标
        if (error < center_tolerance)
        {
            ROS_INFO("Center adjustment complete (%.2f)", current_center_x_);
            center_adjustment_complete = true;
            break;
        }

        // 计算并发布速度指令（只调整y方向）
        double y_vel = calculateYVelocity();
        geometry_msgs::Twist cmd_vel;
        cmd_vel.linear.x = 0;  // x方向不动
        cmd_vel.linear.y = y_vel;
        cmd_vel.angular.z = 0;
        cmd_vel_pub__.publish(cmd_vel);

        center_adjust_count++;
    }

    // 停止运动
    geometry_msgs::Twist stop_vel;
    stop_vel.linear.y = 0;
    cmd_vel_pub__.publish(stop_vel);

    // 超时处理
    if (!center_adjustment_complete)
    {
        ROS_WARN("Center adjustment timed out after %d attempts. Final error: %.2f",
                 max_center_adjust_count, fabs(target_center_x_ - current_center_x_));
    }

    // 第二阶段：调节距离到0.2米（物品已居中后再调整距离）
    const double distance_tolerance = 0.02; // 允许的误差范围（米）
    const int max_distance_adjust_count = 15000; // 最大调节次数
    int distance_adjust_count = 0;
    bool distance_adjustment_complete = false;

    // 重置距离PID参数
    pid_distance_.err = 0;
    pid_distance_.err_last = 0;
    pid_distance_.integral = 0;
    pid_distance_.output = 0;

    ROS_INFO("=== Starting distance adjustment to 0.2m ===");
    
    while (distance_adjust_count < max_distance_adjust_count && !distance_adjustment_complete && ros::ok())
    {
        // 获取当前前方距离
        ROS_INFO("Distance Adjustment %d: Current=%.3fm, Target=%.3fm, Error=%.3fm",
                 distance_adjust_count, distance_qian_x, target_distance_, 
                 fabs(target_distance_ - distance_qian_x));

        // 检查是否达到目标距离
        if (fabs(target_distance_ - distance_qian_x) < distance_tolerance)
        {
            ROS_INFO("Distance adjustment complete (%.3fm)", distance_qian_x);
            distance_adjustment_complete = true;
            break;
        }

        // 计算并发布速度指令（只调整x方向）
        double x_vel = -calculateXVelocity();
        geometry_msgs::Twist cmd_vel;
        cmd_vel.linear.x = x_vel;
        cmd_vel.linear.y = 0;  // y方向不动
        cmd_vel.angular.z = 0;
        cmd_vel_pub__.publish(cmd_vel);

        distance_adjust_count++;
    }

    // 停止运动
    stop_vel.linear.x = 0;
    cmd_vel_pub__.publish(stop_vel);

    // 超时处理
    if (!distance_adjustment_complete)
    {
        ROS_WARN("Distance adjustment timed out after %d attempts. Final error: %.3fm",
                 max_distance_adjust_count, fabs(target_distance_ - distance_qian_x));
    }

    // 获取物品ID
    do
    {
        nh_.getParam("cl", target);
    } while (target != CMD_Apple && target != CMD_Banana && target != CMD_Watermelon &&
             target != CMD_pepper && target != CMD_Tomato && target != CMD_Potato &&
             target != CMD_Milk && target != CMD_Cake && target != CMD_coke);

    ROS_WARN("Target identified: %d", target);
    target1 = target;
    nh_.setParam("target_tool", target);

    // 语音播报
    nh_.setParam("audio", 0);
    play_flag_client.call(_);
    while (!nh_.param("audio", 0))
        ros::spinOnce();
}

// 到达仿真区
void OURSWITCH::GotoC()
{
    ROS_INFO("Entering GotoC state");
    goto_C;
    ac_.waitForResult();
    while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
        ros::spinOnce();
}

// 仿真阶段
void OURSWITCH::Gazebo()
{
    ROS_INFO("Entering Gazebo state");
    std_msgs::Int32 room;

    // 1. 确保room.data被正确初始化（处理所有情况）
    if(target_class == CMD_Fruits)
    {   
        room.data = 1;
    }
    else if(target_class == CMD_Vegetables)  // 用else if避免多重赋值
    {
        room.data = 2;
    }
    else if(target_class == CMD_sweet)
    {
        room.data = 3;
    }
    else
    {
        ROS_ERROR("Invalid target_class: %d, cannot determine room number", target_class);
        return;  // 无效状态下直接返回，避免发布错误数据
    }

    // 2. 持续发布2秒，加入调试信息和回调处理
    ros::Time start_time = ros::Time::now();
    ros::Rate publish_rate(50);  // 10Hz发布频率
    int publish_count = 0;       // 记录发布次数（用于调试）

    while (ros::Time::now() - start_time < ros::Duration(5.0) && ros::ok())
    {
        Gazebo_Command.publish(room);
        publish_count++;
        ROS_INFO("Publishing room: %d (count: %d, remaining: %.2fs)",
                room.data, publish_count,
                2.0 - (ros::Time::now() - start_time).toSec());
        
        publish_rate.sleep();
        ros::spinOnce();  // 处理可能的回调（即使不处理回调，加上更安全）
    }

    ROS_INFO("Finished publishing. Total published: %d times", publish_count);
    
    do
    {
        ROS_INFO("Receiving Room......"); // 房间
    } while (target != CMD_Gazebo1 && target != CMD_Gazebo2 && target != CMD_Gazebo3);
            do
        {
            ROS_INFO("Receiving Food......"); // 记录仿真中取到的物品
        } while (target2 != CMD_Apple && target2 != CMD_Banana && target2 != CMD_Watermelon && target2 != CMD_pepper && target2 != CMD_Tomato && target2 != CMD_Potato && target2 != CMD_Milk && target2 != CMD_Cake && target2 != CMD_coke);

    
    nh_.setParam("target_tool", target);
    ros::Duration(20).sleep();
    // 语音播报
    nh_.setParam("audio", 0);
    play_flag_client.call(_);
    while (!nh_.param("audio", 0))
        ros::spinOnce();
}

// 到达1号标牌前
void OURSWITCH::GotoD()
{
    ROS_INFO("Entering GotoD state");
    // goto_D0;
    // while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
    //     ros::spinOnce();
    goto_D1;
    while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
        ros::spinOnce();

    // 到达D1后等待2秒
    ROS_INFO("Reached D1, waiting 2 seconds...");
    delayedFunction(2);  // 等待2秒

    // 检测颜色信号（1=绿色，2=红色）
    current_color_ = 0;  // 重置颜色状态
    const double timeout = 30.0;  // 30秒超时
    ros::Time start_time = ros::Time::now();
    
    while (current_color_ == 0 && (ros::Time::now() - start_time).toSec() < timeout && ros::ok())
    {   
        ROS_INFO("Waiting for color detection result...");
        ros::spinOnce();
    }

    // 重置参数服务器中路口检测相关参数
    nh_.setParam("/vision_getIntersection1", 0);
    nh_.setParam("/vision_getIntersection2", 0);

    // 处理颜色识别结果
    if (current_color_ == 1)  // 绿色
    {
        vision_getIntersection1 = 1;
        vision_getIntersection2 = 0;
        nh_.setParam("target_tool", CMD_Intersection1);
        // 设置参数服务器值
        nh_.setParam("/vision_getIntersection1", 1);
        ROS_INFO("Detected green light (1), set /vision_getIntersection1=1");
    }
    else if (current_color_ == 2)  // 红色
    {
        vision_getIntersection1 = 0;
        vision_getIntersection2 = 1;
        nh_.setParam("target_tool", CMD_Intersection2);
        // 设置参数服务器值
        nh_.setParam("/vision_getIntersection2", 1);
        ROS_INFO("Detected red light (2), set /vision_getIntersection2=1");
    }
    else  // 超时或无效值
    {
        ROS_WARN("No valid color detected, using default (green)");
        vision_getIntersection1 = 1;
        target = CMD_Intersection1;
        // 设置默认参数服务器值
        nh_.setParam("/vision_getIntersection1", 1);
    }

    // 根据颜色信号导航
    if (vision_getIntersection1 == 1)
    {   
        ROS_INFO("Proceeding through green light");
        
        // 语音播报
        nh_.setParam("audio", 0);
        play_flag_client.call(_);
        while (!nh_.param("audio", 0))
            ros::spinOnce();

        goto_E1;
        while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
            ros::spinOnce();
    }
    else if (vision_getIntersection2 == 1)
    {
        ROS_INFO("Waiting for red light, then proceeding");
        goto_D2;
        ROS_INFO("reaching D2");
        while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
            ros::spinOnce();
        ROS_INFO("reached D2");

        
        nh_.setParam("audio", 0);
        play_flag_client.call(_);
        while (!nh_.param("audio", 0))
            ros::spinOnce();
        ROS_INFO("yuyin");
            
        goto_E2;
        while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
            ros::spinOnce();
    }
}

// 视觉巡线
void OURSWITCH::vision_line()
{
    ROS_INFO("Entering vision_line state");

    // Stop current navigation
    cancel_pub.publish(cancel_msg);
    ros::Duration(0.5).sleep();

    // Initialize parameters
    double target_x, target_y, target_yaw;
    double target_side_dist, target_back_dist;
    int circle_direction = 1;
    int side_flag = 0;  // Lateral direction flag (1: right, -1: left)

    // Set target parameters (distinguish between two intersections)
    if (vision_getIntersection1 == 1)
    {
        target_x = 4.0;
        target_y = 1.23;
        target_yaw = -72.0 * M_PI / 180.0;  // Target heading angle (radians)
        circle_direction = 1;               // Clockwise rotation
        side_flag = 1;                      // Lateral adjustment references right distance
        target_side_dist = 0.25;              // Target lateral distance
        target_back_dist = 1.36;           // Target backward distance
    }
    else if (vision_getIntersection2 == 1)
    {
        target_x = 4.5;
        target_y = 1.23;
        target_yaw = -108.0 * M_PI / 180.0; // Target heading angle (radians)
        circle_direction = -1;              // Counterclockwise rotation
        side_flag = -1;                     // Lateral adjustment references left distance
        target_side_dist = 0.25;            // Target lateral distance
        target_back_dist = 1.36;           // Target backward distance
    }
    else
    {
        ROS_ERROR("No valid intersection detected");
        return;
    }

    // Initialize PID controllers (shared across global stages)
    PID pid_side = {0.3, 0.01, 0.001, 0, 0, 0, 0};         // Lateral distance PID
    PID pid_back = {0.3, 0.01, 0.001, 0, 0, 0, 0};         // Backward distance PID
    PID pid_angular_z0 = {1.0, 0.8, 0.02, 0, 0, 0, 0};
    PID pid_angular_z = {0.3, 0.07, 0.02, 0, 0, 0, 0};     // Heading angle PID


    // Control parameters (global constants)
    const double max_side_vel = 0.3;       // Maximum lateral velocity
    const double max_back_vel = 0.3;       // Maximum forward/backward velocity
    const double distance_tolerance = 0.03; // Distance control tolerance (meters)
    const double yaw_tolerance = 0.017;     // Heading angle control tolerance (0.05=3)

    // State machine definition
    enum Stage { 
        DISTANCE_ADJUST,   // Initial distance adjustment
        PID_CONTROL,       // Precise position and heading control
        CIRCLE_MOTION,     // Circular motion (with heading check)
        YAW_ADJUST0,       // 转完小圈后校准一下
        DISTANCE_ADJUST2,  // Secondary distance adjustment
        YAW_ADJUST,        // Final heading adjustment
        COMPLETE           // Completion
    } current_stage = DISTANCE_ADJUST;
    
    // Global state variables (retained across stages)
    bool side_adjust_complete = false;  // Initial lateral adjustment completion flag
    bool yaw_adjust_complete = false;   // Initial heading adjustment completion flag
    bool side2_complete = false;  // Local state flag (only valid for current stage)
    ros::Time circle_start_time;        // Circle motion start time
    int yaw_count = 0;                  // Heading达标计数
    const double circle_max_duration = 6.0;  // Maximum circle time (seconds)

    // Debug counter (controls print frequency)
    int loop_count = 0;

    ros::Rate loop_rate(50);  // Control loop frequency (50Hz)
    while (ros::ok() && current_stage != COMPLETE)
    {
        loop_count++;

        // Get current state (position, heading, sensor data)
        double current_x = nh_.param("CarX", 0.0);
        double current_y = nh_.param("CarY", 0.0);
        double current_yaw = nh_.param("CarYaw", 0.0);
        double dist_right = distance_you_y;  // Right distance
        double dist_left = distance_zuo_y;   // Left distance
        double dist_back = distance_hou_x;   // Backward distance
        double dist_front = distance_qian_x; // Forward distance

        // Normalize heading angle to [-π, π]
        if (current_yaw > M_PI) current_yaw -= 2 * M_PI;
        if (current_yaw < -M_PI) current_yaw += 2 * M_PI;

        // State machine logic
        switch (current_stage)
        {
        case DISTANCE_ADJUST:
        {
            // Select current lateral distance (based on side_flag)
            double current_side_dist = (side_flag == 1) ? dist_right : dist_left;
            // Calculate lateral error (with direction correction)
            double side_err = (target_side_dist - current_side_dist) * side_flag;
            // Calculate backward error
            double back_err = target_back_dist - dist_back;

            double y_vel = 0.0, x_vel = 0.0;

            if (!side_adjust_complete)
            {
                // Lateral distance adjustment
                y_vel = PID_Realize(&pid_side, side_err, max_side_vel, -max_side_vel);
                
                // Debug info (print every 10 cycles)
                if (loop_count % 10 == 0) {
                    ROS_INFO("[DISTANCE_ADJUST] Lateral velocity: %.3f m/s", y_vel);
                    ROS_INFO("[DISTANCE_ADJUST] Current: %.3f, Target: %.3f, Tolerance: %.3f",
                             current_side_dist, target_side_dist, distance_tolerance);
                }

                // Check if lateral adjustment is complete
                if (fabs(target_side_dist - current_side_dist) <= distance_tolerance)
                {
                    y_vel = 0.0;
                    side_adjust_complete = true;
                    ROS_INFO("Lateral distance adjustment completed");
                }
            }
            else
            {
                // Backward distance adjustment
                x_vel = PID_Realize(&pid_back, back_err, max_back_vel, -max_back_vel);
                
                // Debug info
                if (loop_count % 10 == 0) {
                    ROS_INFO("[DISTANCE_ADJUST] Forward/backward velocity: %.3f m/s", x_vel);
                    ROS_INFO("[DISTANCE_ADJUST] Current: %.3f, Target: %.3f, Tolerance: %.3f",
                             dist_back, target_back_dist, distance_tolerance);
                }

                // Check if backward adjustment is complete
                if (fabs(back_err) <= distance_tolerance)
                {
                    current_stage = PID_CONTROL;
                    x_vel = 0.0;
                    ROS_INFO("Backward distance adjustment completed, entering PID_CONTROL stage");
                }
            }

            // Publish velocity command
            cmd_vel.linear.x = x_vel;
            cmd_vel.linear.y = y_vel;
            cmd_vel.angular.z = 0;
            cmd_vel_pub__.publish(cmd_vel);
            break;
        }

        case PID_CONTROL:
        {
            if (!yaw_adjust_complete)
            {
                // Disable vision control, focus on heading adjustment
                nh_.setParam("/start_vision1", 0);
                vision_control_active_ = false;

                // Calculate heading error (with normalization)
                double yaw_err = target_yaw - current_yaw;
                if (yaw_err > M_PI) yaw_err -= 2 * M_PI;
                if (yaw_err < -M_PI) yaw_err += 2 * M_PI;
                
                // Calculate rotation velocity
                double angular_z = PID_Realize2(&pid_angular_z, yaw_err, 0.1, -0.1, 10);
                
                // Debug info
                if (loop_count % 10 == 0) {
                    ROS_INFO("[PID_CONTROL] Rotation velocity: %.3f rad/s", angular_z);
                    ROS_INFO("[PID_CONTROL] Current heading: %.3f°, Target: %.3f°, Tolerance: %.3f°",
                             current_yaw * 180/M_PI, target_yaw * 180/M_PI,
                             yaw_tolerance * 180/M_PI);
                }

                // Check if heading adjustment is complete
                if (fabs(yaw_err) < yaw_tolerance)
                {
                    angular_z = 0.0;
                    yaw_adjust_complete = true;
                    ROS_INFO("Heading adjustment completed");
                }

                // Publish rotation command
                cmd_vel.linear.x = 0.0;
                cmd_vel.angular.z = angular_z;
                cmd_vel_pub__.publish(cmd_vel);
            }
            else
            {
                // Activate vision control to adjust Y position
                if (!vision_control_active_)
                {
                    nh_.setParam("/start_vision1", 1);
                    vision_control_active_ = true;
                    nh_.setParam("/target_y", target_y);
                }

                // Check Y position error
                double dy = target_y - current_y;
                if (loop_count % 10 == 0) {
                    ROS_INFO("[PID_CONTROL] Y position error: %.3f", dy);
                    ROS_INFO("[PID_CONTROL] Current Y: %.3f, Target Y: %.3f, Tolerance: 0.05",
                             current_y, target_y);
                }

                // Enter circle phase after position is reached
                if (fabs(dy) < 0.05)
                {
                    nh_.setParam("/start_vision1", 0);
                    current_stage = CIRCLE_MOTION;
                    vision_control_active_ = false;
                    cmd_vel.linear.x = 0.0;
                    cmd_vel.angular.z = 0;
                    cmd_vel_pub__.publish(cmd_vel);
                    circle_start_time = ros::Time::now();  // Record circle start time
                    ROS_INFO("Entering CIRCLE_MOTION stage");
                }
            }
            break;
        }

        case CIRCLE_MOTION:
        {
            // Set target heading angle during circle motion (0 degrees or 180 degrees, depending on intersection)
            double target_check_yaw = (vision_getIntersection1 == 1) ? 0 : M_PI;
            
            // Calculate heading error
            double yaw_err = target_check_yaw - current_yaw;
            if (yaw_err > M_PI) yaw_err -= 2 * M_PI;
            if (yaw_err < -M_PI) yaw_err += 2 * M_PI;

            // Check if heading is reached
            bool yaw_reached = fabs(yaw_err) < yaw_tolerance;
            // bool distance_reached = 0;
            
            // Debug info
            if (loop_count % 10 == 0) {
                ROS_INFO("[CIRCLE_MOTION] Current heading: %.3f°, Target: %.3f°, Tolerance: %.3f°",
                         current_yaw * 180/M_PI, target_check_yaw * 180/M_PI,
                         yaw_tolerance * 180/M_PI);
                ROS_INFO("[CIRCLE_MOTION] Heading reached: %s, Count: %d",
                         yaw_reached ? "Yes" : "No", yaw_count);
            }

            // if(vision_getIntersection1 = 1)
            // {
            //     if(distance_hou_x = 1.24)
            //     {
            //         distance_reached = 1;
            //     }
            // }
            // if(vision_getIntersection2 = 1)
            // {
            //     if(distance_qian_x = 1.24)
            //     {
            //         distance_reached = 1;
            //     }
            // }

            // Check if circle phase should exit
            bool time_exceeded = (ros::Time::now() - circle_start_time).toSec() >= circle_max_duration;
            
            if ((yaw_reached || time_exceeded ) && yaw_count != 1)
            {
                yaw_count++;
                ROS_INFO("Heading reached (%.2f°), Count: %d", current_yaw * 180 / M_PI, yaw_count);
            }
            
            bool yaw_condition_met = (yaw_count >= 1);

            if (yaw_condition_met)
            {
                // Stop circle motion
                cmd_vel.linear.x = 0;
                cmd_vel.angular.z = 0;
                cmd_vel_pub__.publish(cmd_vel);
                
                // Output transition reason
                if (yaw_condition_met)
                {
                    ROS_INFO("Heading condition satisfied, entering DISTANCE_ADJUST2 stage");
                }
                else
                {
                    ROS_WARN("Circle motion timed out, entering DISTANCE_ADJUST2 stage");
                }
                
                current_stage = YAW_ADJUST0;
                break;
            }

            // Continue circle motion (fixed radius and linear velocity)
            const double circle_radius = 0.42;  // Circle radius
            const double circle_linear = 0.15;   // Linear velocity
            double circle_angular = circle_linear / circle_radius * circle_direction;  // Angular velocity

            cmd_vel.linear.x = circle_linear;
            cmd_vel.angular.z = circle_angular;
            cmd_vel_pub__.publish(cmd_vel);
            
            // Circle motion status (print every 50 cycles)
            if (loop_count % 50 == 0) {
                ROS_INFO("[CIRCLE_MOTION] Linear velocity: %.3f m/s, Angular velocity: %.3f rad/s",
                         circle_linear, circle_angular);
            }
            break;
        }

        case YAW_ADJUST0:
        {
            // Final target heading angle: -90 degrees (converted to radians)
            double target_final_yaw = (vision_getIntersection1 == 1) ? 0 : M_PI;
            // Calculate heading error (with normalization)
            double yaw_err = target_final_yaw - current_yaw;
            if (yaw_err > M_PI) yaw_err -= 2 * M_PI;
            if (yaw_err < -M_PI) yaw_err += 2 * M_PI;

            // PID-controlled rotation velocity
            double angular_z = PID_Realize2(&pid_angular_z0, yaw_err, 0.1, -0.1, 10);

            // Debug info
            if (loop_count % 10 == 0) {
            ROS_INFO("[YAW_ADJUST0] Rotation velocity: %.3f rad/s", angular_z);
            ROS_INFO("[YAW_ADJUST0] Current heading: %.3f°, Target: %.3f°, Tolerance: %.3f°",
            current_yaw * 180/M_PI, target_final_yaw * 180/M_PI,
            yaw_tolerance * 180/M_PI);
            }

            // Check if final heading adjustment is complete
            if (fabs(yaw_err) < yaw_tolerance)
            {
            angular_z = 0.0;
            current_stage = DISTANCE_ADJUST2;
            ROS_INFO("Final heading adjusted to -90 degrees successfully");
            }

            // Publish rotation command
            cmd_vel.linear.x = 0.0;
            cmd_vel.angular.z = angular_z;
            cmd_vel_pub__.publish(cmd_vel);
            break;
        }

        case DISTANCE_ADJUST2:
        {
            // Local PID controllers (reset each time stage is entered)
            PID pid_side2 = {1.0, 0.04, 0.01, 0, 0, 0, 0};  // Secondary lateral PID
            PID pid_dist2 = {0.5, 0.2, 0.01, 0, 0, 0, 0};  // Secondary forward/backward PID
            
            
            // Target parameters and direction flags (based on intersection)
            double target_side, current_side, target_dist, current_dist;
            int dist_flag = 1;  // Forward/backward direction flag (independent of lateral)
            
            if (vision_getIntersection1 == 1)
            {
                target_side = 1.93;    // Target left distance
                current_side = dist_left;
                target_dist = 1.35;    // Target backward distance
                current_dist = dist_back;
                dist_flag = 1;       // Backward distance adjustment direction correction
            }
            else
            {
                target_side = 1.93;    // Target right distance
                current_side = dist_right;
                target_dist = 1.35;    // Target forward distance
                current_dist = dist_front;
                dist_flag = -1;        // Forward distance adjustment direction correction
            }

            // Initialize velocity command
            cmd_vel.linear.x = 0;
            cmd_vel.linear.y = 0;
            cmd_vel.angular.z = 0;

            if (!side2_complete)
            {
                // Secondary lateral distance adjustment
                double side_err = (target_side - current_side) * -side_flag;  // With direction correction
                cmd_vel.linear.y = PID_Realize(&pid_side2, side_err, max_side_vel, -max_side_vel);
                
                // Debug info
                if (loop_count % 10 == 0) {
                    ROS_INFO("[DISTANCE_ADJUST2] Lateral velocity: %.3f m/s", cmd_vel.linear.y);
                    ROS_INFO("[DISTANCE_ADJUST2] Current lateral distance: %.3f, Target: %.3f, Tolerance: %.3f",
                             current_side, target_side, distance_tolerance);
                }

                // Check if lateral adjustment is complete
                if (fabs(target_side - current_side) <= distance_tolerance)
                {
                    cmd_vel.linear.y = 0;
                    side2_complete = true;
                    ROS_INFO("Secondary lateral distance adjustment completed");
                }
            }
            else
            {
                // Secondary forward/backward distance adjustment
                double dist_err = (target_dist - current_dist) * dist_flag;  // With direction correction
                cmd_vel.linear.x = PID_Realize(&pid_dist2, dist_err, max_back_vel, -max_back_vel);
                
                // Debug info
                if (loop_count % 10 == 0) {
                    ROS_INFO("[DISTANCE_ADJUST2] Forward/backward velocity: %.3f m/s", cmd_vel.linear.x);
                    ROS_INFO("[DISTANCE_ADJUST2] Current: %.3f, Target: %.3f, Tolerance: %.3f",
                             current_dist, target_dist, distance_tolerance);
                }

                // Check if forward/backward adjustment is complete
                if (fabs(target_dist - current_dist) <= distance_tolerance)
                {
                    cmd_vel.linear.x =0;
                    current_stage = YAW_ADJUST;
                    side2_complete = false; // Reset flag
                    ROS_INFO("Secondary distance adjustment completed, entering YAW_ADJUST stage");
                }
            }

                    // Publish velocity command
            cmd_vel_pub__.publish(cmd_vel);
            break;
        }

            case YAW_ADJUST:
            {
            // Final target heading angle: -90 degrees (converted to radians)
            double target_final_yaw = -M_PI / 2;

            // Calculate heading error (with normalization)
            double yaw_err = target_final_yaw - current_yaw;
            if (yaw_err > M_PI) yaw_err -= 2 * M_PI;
            if (yaw_err < -M_PI) yaw_err += 2 * M_PI;

            // PID-controlled rotation velocity
            double angular_z = PID_Realize2(&pid_angular_z, yaw_err, 0.1, -0.1, 10);

            // Debug info
            if (loop_count % 10 == 0) {
            ROS_INFO("[YAW_ADJUST] Rotation velocity: %.3f rad/s", angular_z);
            ROS_INFO("[YAW_ADJUST] Current heading: %.3f°, Target: %.3f°, Tolerance: %.3f°",
            current_yaw * 180/M_PI, target_final_yaw * 180/M_PI,
            yaw_tolerance * 180/M_PI);
            }

            // Check if final heading adjustment is complete
            if (fabs(yaw_err) < yaw_tolerance)
            {
            angular_z = 0.0;
            current_stage = COMPLETE;
            ROS_INFO("Final heading adjusted to -90 degrees successfully");
            }

            // Publish rotation command
            cmd_vel.linear.x = 0.0;
            cmd_vel.angular.z = angular_z;
            cmd_vel_pub__.publish(cmd_vel);
            break;
            }

            case COMPLETE:
            break;
            }

            ros::spinOnce();
            loop_rate.sleep();
            }

            // Reset intersection detection flags
            vision_getIntersection1 = 0;
            vision_getIntersection2 = 0;
            // Clear parameter server values
            nh_.setParam("/vision_getIntersection1", 0);
            nh_.setParam("/vision_getIntersection2", 0);
            ROS_INFO("vision_line state complete");
}

// 到达终点（含避障）
void OURSWITCH::GotoF()
{
    ROS_INFO("Entering GotoF state");

    // 停止当前导航
    cancel_pub.publish(cancel_msg);
    ros::Duration(0.5).sleep();

    // 初始化参数
    const double target_side_dist = 1.26;
    const double target_front_dist = 0.30;
    const double side_tolerance = 0.03;
    const double front_tolerance = 0.08;
    const double yaw_tolerance = 0.03; 
    const double max_y_vel = 0.1;
    const double max_x_vel = 0.1;

    // PID控制器
    PID pid_side = {0.3, 0.08, 0.01, 0, 0, 0, 0};
    PID pid_front = {0.3, 0.02, 0.01, 0, 0, 0, 0};
    PID pid_angular_z = {1.0, 0.8, 0.02, 0, 0, 0, 0};

    // 状态机
    enum Stage { SIDE_ADJUST, FRONT_ADJUST, NAVIGATE_GOAL,LINE, LINE2,LINE3 ,COMPLETE } current_stage = SIDE_ADJUST;
    double current_x = 0.0, current_y = 0.0;

    int loop_count = 0;
    ros::Rate loop_rate(50);
    while (ros::ok() && current_stage != COMPLETE)
    {
        loop_count++;
        // 获取传感器数据
        double current_side = distance_you_y;
        double current_front = distance_qian_x;
        double current_yaw = nh_.param("CarYaw", 0.0);

        switch (current_stage)
        {
        case SIDE_ADJUST:
        {
            double side_err = target_side_dist - current_side;
            double y_vel = PID_Realize2(&pid_side, side_err, max_y_vel, -max_y_vel, 0.5);
            cmd_vel.linear.y = y_vel;

            if (fabs(target_side_dist - current_side) <= side_tolerance)
            {
                cmd_vel.linear.y = 0;
                current_stage = FRONT_ADJUST;
                ROS_INFO("Side adjustment complete");
            }
            cmd_vel.linear.x = 0.0;
            cmd_vel.angular.z = 0.0; 
            cmd_vel_pub__.publish(cmd_vel);
            break;
        }

        case FRONT_ADJUST:
        {
            if (!vision_control_active2_)
            {
                nh_.setParam("/start_vision2", 1);
                vision_control_active2_ = true;
                ROS_INFO("start");
            }

            double front_err = target_front_dist - current_front;

            if (fabs(front_err) <= front_tolerance)
            {
                nh_.setParam("/start_vision2", 0);
                current_stage = NAVIGATE_GOAL;
                vision_control_active2_ = false;
                cmd_vel.linear.x = 0;
                cmd_vel_pub__.publish(cmd_vel);
                ROS_INFO("Front adjustment complete");
            }
        break;
        }

        case NAVIGATE_GOAL:
        {
           ROS_INFO("Navigating to final goal");
            // goto_F;
            // while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
            //     ros::spinOnce();

            ros::Time rotate_start1 = ros::Time::now();
            while (ros::Time::now() - rotate_start1 < ros::Duration(4))
            {
                cancel_pub.publish(cancel_msg);
                cmd_vel.linear.x = 0.0;
                cmd_vel.linear.y = -0.2;
                cmd_vel.angular.z = 0.0;  
                cmd_vel_pub__.publish(cmd_vel);
                ros::spinOnce();
            }

            ros::Time rotate_start2 = ros::Time::now();
            while (ros::Time::now() - rotate_start2 < ros::Duration(3.5))
            {
                cancel_pub.publish(cancel_msg);
                cmd_vel.linear.x = 0.3;
                cmd_vel.linear.y = 0.0;
                cmd_vel.angular.z = 0.0;  
                cmd_vel_pub__.publish(cmd_vel);
                ros::spinOnce();
            }
            cmd_vel.linear.x = 0.0;
            cmd_vel.linear.y = 0.0;
            cmd_vel.angular.z = 0.0; 
            cmd_vel_pub__.publish(cmd_vel);
            
            // 停止移动
            cmd_vel.linear.x = 0.0;
            cmd_vel_pub__.publish(cmd_vel);
                
            current_stage = LINE;
            break;
	}
    case LINE:
    {
        // Local PID controllers (reset each time stage is entered)
        PID pid_side3 = {1.0, 0.04, 0.01, 0, 0, 0, 0};  
        // Target parameters and direction flags (based on intersection)
        double target_side, current_side;
        const double max_side_vel = 0.3;
        const double distance_tolerance = 0.03;
        target_side = 1.35;    // Target right distance
        current_side = distance_you_y;

        // Initialize velocity command
        cmd_vel.linear.x = 0;
        cmd_vel.linear.y = 0;
        cmd_vel.angular.z = 0;

        // Secondary lateral distance adjustment
        double side_err = (target_side - current_side);
        cmd_vel.linear.y = PID_Realize(&pid_side3, side_err, max_side_vel, -max_side_vel);
        
        // Debug info
        if (loop_count % 10 == 0) {
            ROS_INFO("[LINE] Lateral velocity: %.3f m/s", cmd_vel.linear.y);
            ROS_INFO("[LINE] Current lateral distance: %.3f, Target: %.3f, Tolerance: %.3f",
                        current_side, target_side, distance_tolerance);
        }

        // Check if lateral adjustment is complete
        if (fabs(target_side - current_side) <= distance_tolerance)
        {   
            current_stage = LINE2;
            cmd_vel.linear.y = 0;
            ROS_INFO("LINE completed");
        }
        
        cmd_vel_pub__.publish(cmd_vel);
        break;
    }

    case LINE2:
    {
        // Final target heading angle: -90 degrees (converted to radians)
        
        double target_final_yaw = -M_PI / 2;

        // Calculate heading error (with normalization)
        double yaw_err = target_final_yaw - current_yaw;
        if (yaw_err > M_PI) yaw_err -= 2 * M_PI;
        if (yaw_err < -M_PI) yaw_err += 2 * M_PI;

        // PID-controlled rotation velocity
        double angular_z = PID_Realize(&pid_angular_z, yaw_err, 0.1, -0.1);

        // Debug info
        if (loop_count % 10 == 0) {
        ROS_INFO("[YAW_ADJUST] Rotation velocity: %.3f rad/s", angular_z);
        ROS_INFO("[YAW_ADJUST] Current heading: %.3f°, Target: %.3f°, Tolerance: %.3f°",
        current_yaw * 180/M_PI, target_final_yaw * 180/M_PI,
        yaw_tolerance * 180/M_PI);
        }

        // Check if final heading adjustment is complete
        if (fabs(yaw_err) < yaw_tolerance)
        {
        angular_z = 0.0;
        current_stage = LINE3;
        ROS_INFO("Final heading adjusted to -90 degrees successfully");
        }

        // Publish rotation command
        cmd_vel.linear.x = 0.0;
        cmd_vel.angular.z = angular_z;
        cmd_vel_pub__.publish(cmd_vel);
        break;
    }
    case LINE3:
    {
        goto_F;
        ac_.waitForResult();
        while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED))
            ros::spinOnce();
        current_stage = COMPLETE;
        break;
    }

    case COMPLETE:
        break;
    }

        ros::spinOnce();
        loop_rate.sleep();
    }

    ROS_INFO("GotoF state complete");
}

// 结束阶段
void OURSWITCH::END()
{
    ROS_INFO("Entering END state");
    target3 = target1 + target2;
    nh_.setParam("target_tool", target3);

    // 语音播报最终结果
    nh_.setParam("audio", 0);
    play_flag_client.call(_);
    while (!nh_.param("audio", 0))
        ros::spinOnce();
}

// 二维码回调函数
void OURSWITCH::getZbarCallback(const std_msgs::String::ConstPtr &msg)
{
    std::string received_msg = msg->data;
    if(Zbar_flag == 1)
    {
        if (received_msg == "Fruit") {
        nh_.setParam("target_class", CMD_Fruits);
        Zbar_flag = 0;
        // ROS_INFO("Received: Fruit");
        } 
        else if (received_msg == "Vegetable") {
            nh_.setParam("target_class", CMD_Vegetables);
            Zbar_flag = 0;
            // ROS_INFO("Received: Vegetable");
        } 
        else if (received_msg == "Dessert") {
            nh_.setParam("target_class", CMD_sweet);
            Zbar_flag = 0;
            // ROS_INFO("Received: Dessert");
        } 
        else {
            ROS_WARN("Received unknown message: %s", received_msg.c_str());
        }
    }
    
}

// 视觉角度回调
void OURSWITCH::visionAngleCallback(const std_msgs::Float32MultiArray::ConstPtr& msg)
{
    if (msg->data.size() >= 3)
    {
        vision_X = msg->data[0];
        vision_Y = msg->data[1];
        vision_Angle = msg->data[2];
    }
    else
    {
        ROS_WARN("Incomplete vision angle data (size: %zu)", msg->data.size());
    }
}

// 仿真房间号回调
void OURSWITCH::getGzeboRoomCallback(const std_msgs::Int32::ConstPtr &msg) 
{
    if(gazebo_flag = 1)
    {
        target = msg->data;
    }
    
}

// 仿真食物号回调
void OURSWITCH::getGzeboFoodCallback(const std_msgs::Int32::ConstPtr &msg) 
{
    target2 = msg->data;
    // ROS_INFO("Received food number: %d", target2);
}

// 超声波数据回调
void OURSWITCH::UltrasoundCallback(const pcl_work::ultrasoundConstPtr &msg)
{
    distance_qian_x = msg->distance_qian_x; 
    distance_zuo_y = msg->distance_zuo_y;
    distance_hou_x = -msg->distance_hou_x;
    distance_you_y = -msg->distance_you_y;
}

// 新增：颜色识别结果回调函数
void OURSWITCH::colorResultCallback(const std_msgs::Int32::ConstPtr& msg)
{
    current_color_ = msg->data;
    // ROS_INFO("Received color result: %d (1=green, 2=red)", current_color_);
    
    // 验证结果有效性
    if (current_color_ != 1 && current_color_ != 2)
    {
        // ROS_WARN("Invalid color value: %d (expected 1 or 2)", current_color_);
        current_color_ = 0; // 标记为无效
    }
}

// 主函数
int main(int argc, char **argv)
{
    ros::init(argc, argv, "switch_node");
    ros::NodeHandle nh;
    
    OURSWITCH ucar;
    ros::AsyncSpinner spinner(1);     
    spinner.start();

    ros::Rate loop_rate(50);
    while (ros::ok())
    {   
    	
        switch (ucar.current_state)
        {
        case TEST_:
            break;
        case GOTOA_:
            ucar.GotoA();  
            ucar.current_state = GOTOB_;
            break;
        case GOTOB_:
            ucar.GotoB();
            break;
        case VISION_GETTOOL_:
            ucar.Vision_GetTool();
            ucar.current_state = GOTOC_;
            break;
        case GOTOC_:
            ucar.GotoC();
            ucar.current_state = Gazebo_;
            break;
        case Gazebo_:
            ucar.Gazebo();
            ucar.current_state = GOTOD_;   
            break;
        case GOTOD_:
            ucar.GotoD();
            ucar.current_state = VISION_LINE_;
            break;
        case VISION_LINE_:
            ucar.vision_line();
            ucar.current_state = GOTOF_;
            break;
        case GOTOF_:
            ucar.GotoF();
            ucar.current_state = END_;
            break;
        case END_:
            ucar.END();
            ucar.current_state = TEST_;
            break;
        }

        loop_rate.sleep();
    }
    
    spinner.stop();
    return 0;
}
    
