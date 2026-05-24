#include "switch2.h"
#include <thread>
#include <chrono>

// 导航点宏定义
#define goto_B sendPos(-1.56, -0.7, 3.14)
#define goto_D sendPos(3.7, 4.0, 0.0)

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseAction;

// =========================================================================
// 构造与析构
// =========================================================================
OURSWITCH::OURSWITCH()
    : ac_(nh_, "move_base", true)
{
    ROS_INFO("switch node initialized");
    current_state = GOTOA_;

    nh_.setParam("yolo_begin", 0);
    position_srv.request.ask = true;

    vision_gettool_client_ = nh_.serviceClient<ourgoal::getLaserPoint>("/srv_getLaserPoint");
    cancel_pub = nh_.advertise<actionlib_msgs::GoalID>("move_base/cancel", 10);
    cmd_vel_pub__ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
    Gazebo_Command = nh_.advertise<std_msgs::Int32>("class_command", 10);

    teb_param_reloader = nh_.serviceClient<ourgoal::srv_reload>("/param_reload");
    getPosition_client = nh_.serviceClient<ourgoal::getPosition>("/srv_getPosition");
    sub_ultrasound = nh_.subscribe("/ultra", 10, &OURSWITCH::UltrasoundCallback, this);
    sub_odom_ = nh_.subscribe("/odom", 10, &OURSWITCH::OdomCallback, this);

    // 默认数据初始化
    distance_qian_x = 1.13; 
    distance_zuo_y = 0.25; 
    distance_hou_x = 0.36; 
    distance_you_y = 0.25; 
    yaw = 0.0;
    
    safe_F = 0.15; 
    safe_B = 0.35; 
    safe_L = 0.20; 
    safe_R = 0.20; 
    Kp_dist = 1.0; 
    Kp_yaw = 2.0; 
    max_vel = 0.5; 
    safe_R2 = 0.25;

    ROS_WARN("Initialization complete!");
    ac_.waitForServer(ros::Duration(5));

    // =======================================================
    // 异步协同：等待 AI.py 录音结束的 awake2 信号
    // =======================================================
    ROS_INFO("Waiting for AI.py to finish recording audio...");
    while (!nh_.param("awake2", 0) && ros::ok()) 
    {
        ros::Duration(0.1).sleep();
    }
    ROS_WARN("Awake2 received! System started, rushing out of the maze!");
}

OURSWITCH::~OURSWITCH() 
{ 
    ROS_WARN("switch node terminated"); 
}

// =========================================================================
// 基础回调与数学库函数 (完全展开版)
// =========================================================================
void OURSWITCH::OdomCallback(const nav_msgs::Odometry::ConstPtr &msg) 
{ 
    tf2::Quaternion q(
        msg->pose.pose.orientation.x, 
        msg->pose.pose.orientation.y, 
        msg->pose.pose.orientation.z, 
        msg->pose.pose.orientation.w
    ); 
    tf2::Matrix3x3 m(q); 
    m.getRPY(roll, pitch, yaw); 
}

void OURSWITCH::UltrasoundCallback(const pcl_work::ultrasoundConstPtr &msg) 
{ 
    distance_qian_x = msg->distance_qian_x; 
    distance_zuo_y = msg->distance_zuo_y; 
    distance_hou_x = -msg->distance_hou_x; 
    distance_you_y = -msg->distance_you_y; 
}

void OURSWITCH::delayedFunction(int delayInSeconds) 
{ 
    std::this_thread::sleep_for(std::chrono::seconds(delayInSeconds)); 
}

void OURSWITCH::PrintfArray(float arr[], int size) 
{ 
    for (int i = 0; i < size; i++) 
    {
        printf("Element %d: %f\n", i, arr[i]); 
    }
}

double OURSWITCH::Limit_Value(double INPUT, double MAX, double MIN) 
{ 
    if (INPUT > MAX) return MAX; 
    else if (INPUT < MIN) return MIN; 
    return INPUT; 
}

double OURSWITCH::PID_Realize(PID *pid, double err, double MAX, double MIN) 
{ 
    pid->err = err; 
    pid->integral += pid->err; 
    double output = pid->kp * pid->err + pid->ki * pid->integral + pid->kd * (pid->err - pid->err_last); 
    pid->err_last = pid->err; 
    pid->output = output; 
    return Limit_Value(pid->output, MAX, MIN); 
}

double OURSWITCH::PID_Realize2(PID *pid, double err, double MAX, double MIN, double integral_limit) 
{ 
    pid->err = err; 
    pid->integral += pid->err; 
    
    if (fabs(pid->integral) > integral_limit) 
    {
        pid->integral = (pid->integral > 0) ? integral_limit : -integral_limit; 
    }
    
    double output = pid->kp * pid->err + pid->ki * pid->integral + pid->kd * (pid->err - pid->err_last); 
    pid->err_last = pid->err; 
    pid->output = output; 
    return Limit_Value(pid->output, MAX, MIN); 
}

void OURSWITCH::InitPID() 
{ 
    Aid_X.err = 0; 
    Aid_X.err_last = 0; 
    Aid_X.integral = 0; 
    Aid_X.output = 0; 
    
    nh_.param("Aid_X_kp", Aid_X.kp, 1.0); 
    nh_.param("Aid_X_ki", Aid_X.ki, 0.0); 
    nh_.param("Aid_X_kd", Aid_X.kd, 0.0); 
}

bool OURSWITCH::getCenterXFromParam() 
{ 
    if (nh_.getParam("center_x", current_center_x_)) 
    { 
        if (current_center_x_ < 0 || current_center_x_ > 640) 
        {
            return false; 
        }
        return true; 
    } 
    return false; 
}

double OURSWITCH::calculateYVelocity() 
{ 
    double error = target_center_x_ - current_center_x_; 
    pid_center_x_.integral += error; 
    
    if (pid_center_x_.integral > 200.0) 
        pid_center_x_.integral = 200.0; 
    else if (pid_center_x_.integral < -200.0) 
        pid_center_x_.integral = -200.0; 
        
    double derivative = error - pid_center_x_.err_last; 
    double output = pid_center_x_.kp * error + pid_center_x_.ki * pid_center_x_.integral + pid_center_x_.kd * derivative; 
    pid_center_x_.err_last = error; 
    
    return Limit_Value(output, 0.1, -0.1); 
}

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

point_2d OURSWITCH::translate(point_2d p, double dx, double dy) 
{ 
    p.x += dx; 
    p.y += dy; 
    return p; 
}

point_2d OURSWITCH::rotate(point_2d p, double CarYaw) 
{ 
    double NewPointX = p.x * cos(CarYaw) - p.y * sin(CarYaw); 
    double NewPointY = p.x * sin(CarYaw) + p.y * cos(CarYaw); 
    p.x = NewPointX; 
    p.y = NewPointY; 
    return p; 
}

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

void OURSWITCH::sendPos(double x, double y, double yaw) 
{ 
    move_base_msgs::MoveBaseGoal goal; 
    goal.target_pose.header.stamp = ros::Time::now(); 
    goal.target_pose.header.frame_id = "map"; 
    
    goal.target_pose.pose.position.x = x; 
    goal.target_pose.pose.position.y = y; 
    goal.target_pose.pose.position.z = 0; 
    
    tf2::Quaternion q; 
    q.setRPY(0, 0, yaw); 
    goal.target_pose.pose.orientation = tf2::toMsg(q); 
    
    ac_.sendGoal(goal); 
    ROS_INFO("Sent goal: x=%f, y=%f, yaw=%f", x, y, yaw); 
}

// =========================================================================
// 仓储区任务 (雷达前8阶段平移逃脱 + 提前原地调头)
// =========================================================================
void OURSWITCH::GotoA()
{
    ROS_INFO("Entering GotoA state: Escaping Maze. (AI is processing LLM in parallel!)");
    
    ros::Rate rate(20);
    int escape_state = 1;
    bool task_finished = false;
    int rotate_count = 0; 

    while (ros::ok() && !task_finished)
    {
        ros::spinOnce();
        geometry_msgs::Twist cmd;
        
        // 锁死 yaw 角，防止车体倾斜
        cmd.angular.z = Kp_yaw * (0.0 - yaw);
        double error = 0;

        switch (escape_state)
        {
            case 1: 
                error = distance_qian_x - safe_F; 
                cmd.linear.x = Limit_Value(Kp_dist * error, max_vel, -max_vel); 
                cmd.linear.y = 0; 
                if (std::abs(error) < 0.05) escape_state = 2; 
                break;
                
            case 2: 
                error = distance_you_y - safe_R; 
                cmd.linear.y = -Limit_Value(Kp_dist * error, max_vel, -max_vel); 
                cmd.linear.x = 0; 
                if (std::abs(error) < 0.05) escape_state = 3; 
                break;
                
            case 3: 
                error = distance_qian_x - safe_F; 
                cmd.linear.x = Limit_Value(Kp_dist * error, max_vel, -max_vel); 
                cmd.linear.y = 0; 
                if (std::abs(error) < 0.05) escape_state = 4; 
                break;
                
            case 4: 
                error = distance_zuo_y - safe_L; 
                cmd.linear.y = Limit_Value(Kp_dist * error, max_vel, -max_vel); 
                cmd.linear.x = 0; 
                if (std::abs(error) < 0.05) escape_state = 5; 
                break;
                
            case 5: 
                error = distance_qian_x - safe_F; 
                cmd.linear.x = Limit_Value(Kp_dist * error, max_vel, -max_vel); 
                cmd.linear.y = 0; 
                if (std::abs(error) < 0.05) escape_state = 6; 
                break;
                
            case 6: 
                error = distance_you_y - safe_R; 
                cmd.linear.y = -Limit_Value(Kp_dist * error, max_vel, -max_vel); 
                cmd.linear.x = 0; 
                if (std::abs(error) < 0.05) escape_state = 7; 
                break;
                
            case 7: 
                error = distance_hou_x - safe_B; 
                cmd.linear.x = -Limit_Value(Kp_dist * error, max_vel, -max_vel); 
                cmd.linear.y = 0; 
                if (std::abs(error) < 0.05) escape_state = 8; 
                break;
                
            case 8: 
                error = distance_zuo_y - safe_L; 
                cmd.linear.y = Limit_Value(Kp_dist * error, max_vel, -max_vel); 
                cmd.linear.x = 0; 
                if (std::abs(error) < 0.05) escape_state = 9; 
                break;
                
            case 9: 
                error = distance_hou_x - safe_B; 
                cmd.linear.x = -Limit_Value(Kp_dist * error, max_vel, -max_vel); 
                cmd.linear.y = 0; 
                if (std::abs(error) < 0.05) escape_state = 10; 
                break;
                
            case 10: 
                error = distance_you_y - safe_R2; 
                cmd.linear.y = -Limit_Value(Kp_dist * error, max_vel, -max_vel); 
                cmd.linear.x = 0; 
                if (std::abs(error) < 0.05) escape_state = 11; 
                break;
                
            case 11: 
                // 解除锁头，强行旋转 180 度调头
                cmd.angular.z = 1.57; 
                cmd.linear.x = 0; 
                cmd.linear.y = 0;
                rotate_count++;
                if (rotate_count >= 45) 
                {
                    task_finished = true;
                }
                break;
        }
        
        cmd_vel_pub__.publish(cmd);
        rate.sleep();
    }
    
    // 强制刹车停车
    geometry_msgs::Twist stop;
    cmd_vel_pub__.publish(stop);
    ROS_INFO("GotoA finished. Going to Point B...");
}

/// =========================================================================
// 版本二：物品领取区交接 (持续旋转极限测试版)
// =========================================================================
void OURSWITCH::GotoB()
{
    ROS_INFO("Entering GotoB state: Navigating to Point B");

    goto_B;
    while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED) && ros::ok())
    {
        ros::spinOnce();
    }
    ROS_INFO("Arrived at Point B!");

    // 🚑 到了 B 点必须第一时间通知 AI 开启摄像头或开始结算！
    nh_.setParam("start_qr_scan", 1);

    // ================= 核心战术：检查是否在路上已经扫齐了 =================
    int scan_done = 0;
    nh_.getParam("qr_scan_done", scan_done);

    if (scan_done == 1) 
    {
        ROS_INFO("🏆 PERFECT! All 3 QR codes were already scanned during GotoA!");
    } 
    else 
    {
        ROS_WARN("⚠️ Not all QR codes found yet. Starting [Continuous Spin] search...");
        
        geometry_msgs::Twist spin_cmd;
        ros::Rate r(10); 
        
        // 持续匀速旋转策略
        // 注意：角速度设置得越慢，拖影越小，扫上的概率越大。这里设为 0.3。
        spin_cmd.angular.z = 0.3; 

        while (scan_done == 0 && ros::ok())
        {
            cmd_vel_pub__.publish(spin_cmd);
            
            nh_.getParam("qr_scan_done", scan_done);
            ros::spinOnce();
            r.sleep();
        }

        // 扫齐后彻底刹车
        spin_cmd.angular.z = 0.0;
        cmd_vel_pub__.publish(spin_cmd);
        ROS_INFO("3 QR Codes found! Car stopped.");
    }

    // ================= 等待 AI 播报完毕 =================
    ROS_INFO("Waiting for AI.py to finish LLM matching and Broadcasting...");
    int task1_done = 0;
    ros::Rate wait_rate(10);
    while (task1_done == 0 && ros::ok())
    {
        nh_.getParam("task1_all_done", task1_done);
        ros::spinOnce();
        wait_rate.sleep();
    }

    ROS_INFO("AI.py broadcast completed and self-terminated to save CPU.");
    current_state = XingHuoAI_;
}

// =========================================================================
// 版本一：物品领取区交接 (转停转停防拖影版)
// =========================================================================
// void OURSWITCH::GotoB()
// {
//     ROS_INFO("Entering GotoB state: Navigating to Point B");

//     goto_B;
//     while (!(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED) && ros::ok())
//     {
//         ros::spinOnce();
//     }
//     ROS_INFO("Arrived at Point B!");

//     // 🚑 到了 B 点必须第一时间通知 AI 开启摄像头或开始结算！
//     nh_.setParam("start_qr_scan", 1);

//     // ================= 核心战术：检查是否在路上已经扫齐了 =================
//     int scan_done = 0;
//     nh_.getParam("qr_scan_done", scan_done);

//     if (scan_done == 1) 
//     {
//         ROS_INFO("🏆 PERFECT! All 3 QR codes were already scanned during GotoA!");
//     } 
//     else 
//     {
//         ROS_WARN("⚠️ Not all QR codes found yet. Starting [Stop-and-Go] search...");
        
//         geometry_msgs::Twist spin_cmd;
//         int tick = 0;
//         ros::Rate r(10); // 10Hz，每周期 0.1 秒
        
//         // 走走停停策略寻找剩余的二维码
//         while (scan_done == 0 && ros::ok())
//         {
//             tick++;
//             // 周期为4秒
//             if (tick % 40 < 20) 
//             {
//                 spin_cmd.angular.z = 0.5; // 转动
//             } 
//             else 
//             {
//                 spin_cmd.angular.z = 0.0; // 刹车，给摄像头留出清晰拍照时间
//             }

//             cmd_vel_pub__.publish(spin_cmd);
            
//             nh_.getParam("qr_scan_done", scan_done);
//             ros::spinOnce();
//             r.sleep();
//         }

//         // 扫齐后彻底刹车
//         spin_cmd.angular.z = 0.0;
//         cmd_vel_pub__.publish(spin_cmd);
//         ROS_INFO("3 QR Codes found! Car stopped.");
//     }

//     // ================= 等待 AI 播报完毕 =================
//     ROS_INFO("Waiting for AI.py to finish LLM matching and Broadcasting...");
//     int task1_done = 0;
//     ros::Rate wait_rate(10);
//     while (task1_done == 0 && ros::ok())
//     {
//         nh_.getParam("task1_all_done", task1_done);
//         ros::spinOnce();
//         wait_rate.sleep();
//     }

//     ROS_INFO("AI.py broadcast completed and self-terminated to save CPU.");
//     current_state = XingHuoAI_;
// }

// =========================================================================
// 读取统一规格的参数字典
// =========================================================================
void OURSWITCH::XingHuoAI()
{
    ROS_INFO("Entering XingHuoAI state: Loading parameters from AI.py...");

    std::string real_item = "UNKNOWN", real_class = "UNKNOWN", real_room = "UNKNOWN";
    std::string sim_item = "UNKNOWN", sim_class = "UNKNOWN", sim_room = "UNKNOWN";

    nh_.getParam("real_item", real_item);
    nh_.getParam("real_class", real_class);       
    nh_.getParam("real_room", real_room);

    nh_.getParam("sim_item", sim_item);
    nh_.getParam("sim_class", sim_class);         
    nh_.getParam("sim_room", sim_room);       

    ROS_INFO("=== Standardized Task Params ===");
    ROS_INFO("Real Car: [%s] -> [%s] -> [%s]", real_item.c_str(), real_class.c_str(), real_room.c_str());
    ROS_INFO("Sim  Car: [%s] -> [%s] -> [%s]", sim_item.c_str(), sim_class.c_str(), sim_room.c_str());
    ROS_INFO("================================");
    
    current_state = GOTOC1_; 
}

// =========================================================================
// 抵达实体与仿真观测点 (包含播报2)
// =========================================================================
void OURSWITCH::GotoC(int target_num)
{
    if(target_num == 1) 
    {
        ROS_INFO("Entering GotoC1 state: Real Car Warehouse Matching");
        std::string target_warehouse = "UNKNOWN";
        nh_.getParam("real_room", target_warehouse);
        nh_.setParam("auto_park_target", target_warehouse);
    } 
    else 
    {
        ROS_INFO("Entering GotoC2 state: Sim Car Warehouse Matching");
        std::string target_warehouse = "UNKNOWN";
        nh_.getParam("sim_room", target_warehouse);
        nh_.setParam("auto_park_target", target_warehouse);
    }

    struct Pose { double x, y, z, w; };
    std::vector<Pose> search_points = {
        {2.0, 3.5, 0.707, 0.707},
        {3.5, 3.5, 0.707, 0.707},
        {5.0, 3.5, 0.707, 0.707}
    };

    bool parking_success = false;
    int point_count = 0;

    while (!parking_success && point_count < search_points.size() && ros::ok())
    {
        ROS_INFO("Navigating to observation point %d...", point_count + 1);
        
        // 此处为调试暂时注释了实际寻点代码
        // sendPos(search_points[point_count].x, search_points[point_count].y, 
        //         search_points[point_count].z, search_points[point_count].w);
                
        bool finished_before_timeout = ac_.waitForResult(ros::Duration(20.0));
        
        if (finished_before_timeout && ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
        {
            ROS_INFO("Arrived at point %d. Auto-Park starting...", point_count + 1);
            nh_.setParam("start_auto_park", 1);
            nh_.setParam("auto_park_status", "WAITING"); 
            
            std::string park_status = "WAITING";
            while (park_status == "WAITING" && ros::ok())
            {
                nh_.getParam("auto_park_status", park_status);
                ros::Duration(0.1).sleep();
                ros::spinOnce();
            }

            if (park_status == "DONE") 
            {
                parking_success = true;
            } 
            else 
            {
                point_count++;
            }
            nh_.setParam("start_auto_park", 0);
        }
        else
        {
            if (!finished_before_timeout) 
            {
                ac_.cancelGoal(); 
            }
            point_count++;
        }
    }

    // ========== 语音播报 2：实体车间放置完成 ==========
    if (parking_success && target_num == 1) 
    {
        std::string item, room;
        nh_.getParam("real_item", item);
        nh_.getParam("real_room", room);
        char tts_cmd[512];
        
        // 拼接发音指令：已将香蕉放入食品加工车间
        sprintf(tts_cmd, "espeak -v zh+f2 \"已将%s放入%s\" -s 130", item.c_str(), room.c_str());
        ROS_INFO("Broadcasting Real Warehouse placement...");
        system(tts_cmd); 
    }

    if (!parking_success) 
    {
        ROS_ERROR("Failed to park at targets! Forcing blind parking.");
    }

    if(target_num == 1) 
    {
        current_state = GOTOC2_;
    } 
    else 
    {
        current_state = Gazebo_;
    }
}

// =========================================================================
// 仿真协同 (包含播报3)
// =========================================================================
void OURSWITCH::Gazebo()
{
    ROS_INFO("Entering Gazebo state: Simulation Task Collaboration");

    nh_.setParam("start_gazebo_sim", 1);
    nh_.setParam("gazebo_sim_done", 0);
    
    int sim_done = 0;
    while (sim_done == 0 && ros::ok())
    {
        nh_.getParam("gazebo_sim_done", sim_done);
        ros::Duration(0.1).sleep();
        ros::spinOnce();
    }
    
    ROS_INFO("Gazebo simulation task reported as COMPLETE!");
    nh_.setParam("start_gazebo_sim", 0);

    // ========== 语音播报 3：仿真任务完成 ==========
    std::string sim_item, sim_room;
    nh_.getParam("sim_item", sim_item);
    nh_.getParam("sim_room", sim_room);
    char tts_cmd[512];
    
    // 拼接发音指令：仿真任务已完成，已将毛巾放入电子产品生产车间
    sprintf(tts_cmd, "espeak -v zh+f2 \"仿真任务已完成，已将%s放入%s\" -s 130", sim_item.c_str(), sim_room.c_str());
    ROS_INFO("Broadcasting Gazebo task completion...");
    system(tts_cmd);

    current_state = GOTOD_; 
}

// =========================================================================
// 交通决策与路径选择
// =========================================================================
void OURSWITCH::GotoD()
{
    ROS_INFO("Entering GotoD state: Traffic Light Detection");
    
    goto_D; 
    bool finished_before_timeout = ac_.waitForResult(ros::Duration(20.0));

    if (finished_before_timeout && ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
    {
        ROS_INFO("Arrived at the stop line successfully.");
    }
    else 
    {
        if (!finished_before_timeout) 
        {
            ac_.cancelGoal();
        }
    }

    nh_.setParam("start_traffic_light_det", 1);
    nh_.setParam("traffic_light_result", "WAITING");
    
    std::string tl_result = "WAITING";
    while (tl_result == "WAITING" && ros::ok())
    {
        nh_.getParam("traffic_light_result", tl_result);
        ros::Duration(0.1).sleep();
        ros::spinOnce();
    }
    
    ROS_INFO("Traffic Light Detected: [%s]", tl_result.c_str());
    nh_.setParam("start_traffic_light_det", 0);

    current_state = VISION_LINE_;
}

// =========================================================================
// 视觉巡线到达终点 (包含播报4)
// =========================================================================
void OURSWITCH::vision_line()
{
    ROS_INFO("Entering VISION_LINE state");

    nh_.setParam("start_vision_line", 1);
    nh_.setParam("vision_line_done", 0); 

    int line_done = 0;
    while (line_done == 0 && ros::ok())
    {
        nh_.getParam("vision_line_done", line_done);
        ros::Duration(0.1).sleep();
        ros::spinOnce();
    }
    
    ROS_INFO("Line tracking completed! Car Stopped.");
    nh_.setParam("start_vision_line", 0); 
    
    // ========== 语音播报 4：任务完成 ==========
    ros::Duration(2.0).sleep(); // 停稳后缓冲2秒，满足“停后须在10秒内开始播报”规则
    ROS_INFO("Broadcasting Final Mission Complete...");
    
    // 直接调用 espeak 播报
    system("espeak -v zh+f2 \"任务完成\" -s 130");

    ROS_INFO("ALL TASKS COMPLETED SUCCESSFULLY! SHUTTING DOWN.");
    ros::shutdown(); 
}

// =========================================================================
// 主函数与状态机主循环
// =========================================================================
int main(int argc, char **argv)
{
    ros::init(argc, argv, "switch_node");
    ros::NodeHandle nh;
    
    OURSWITCH ucar;
    ros::AsyncSpinner spinner(1);     
    spinner.start();

    ros::Rate loop_rate(10);
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
            case XingHuoAI_: 
                ucar.XingHuoAI(); 
                break;
            case GOTOC1_:    
                ucar.GotoC(1); 
                break;
            case GOTOC2_:    
                ucar.GotoC(2); 
                break;
            case Gazebo_:    
                ucar.Gazebo(); 
                break;
            case GOTOD_:     
                ucar.GotoD(); 
                break;
            case VISION_LINE_: 
                ucar.vision_line(); 
                break;
        }
        loop_rate.sleep();
    }
    
    spinner.stop();
    return 0;
}