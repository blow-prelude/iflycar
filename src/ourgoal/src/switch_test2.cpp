#include "switch2.h"
#include <thread>
#include <chrono>
#include <limits>
#include <nav_msgs/GetPlan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <ros/topic.h>

// 导航点宏定义
// #define goto_B sendPos(-1.56, -0.5, 3.14)
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
    sub_scan_ = nh_.subscribe("/scan", 10, &OURSWITCH::ScanCallback, this);

    sub_signal_class_ = nh_.subscribe("/signal_class", 10, &OURSWITCH::SignalClassCallback, this);
    sub_signal_detection_ = nh_.subscribe("/signal_detection", 10, &OURSWITCH::SignalDetectionCallback, this);

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

    nh_.param("gap_min_width", gap_min_width_, 0.75);
    nh_.param("gap_max_width", gap_max_width_, 1.10);
    nh_.param("gap_near_max_range", gap_near_max_range_, 1.50);
    nh_.param("gap_search_half_angle", gap_search_half_angle_, 2.35);
    nh_.param("gap_detection_timeout", gap_detection_timeout_, 3.0);
    nh_.param("gap_sample_max_spread", gap_sample_max_spread_, 0.15);
    nh_.param("gap_wall_min_length", gap_wall_min_length_, 0.20);
    nh_.param("gap_wall_max_residual", gap_wall_max_residual_, 0.03);
    nh_.param("gap_wall_max_line_offset", gap_wall_max_line_offset_, 0.08);
    nh_.param("gap_max_lateral_offset", gap_max_lateral_offset_, 0.45);
    nh_.param("gap_min_forward_offset", gap_min_forward_offset_, -0.20);
    nh_.param("gap_max_forward_offset", gap_max_forward_offset_, 0.80);
    nh_.param("lidar_offset_x", lidar_offset_x_, 0.11);
    nh_.param("lidar_offset_y", lidar_offset_y_, 0.0);
    nh_.param("lidar_yaw", lidar_yaw_, -0.07);
    nh_.param("gap_required_samples", gap_required_samples_, 5);
    nh_.param("gap_wall_min_points", gap_wall_min_points_, 6);
    gap_required_samples_ = std::max(3, gap_required_samples_);
    gap_wall_min_points_ = std::max(3, gap_wall_min_points_);

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
        msg->pose.pose.orientation.w);
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

bool OURSWITCH::detectGap(const sensor_msgs::LaserScan &scan,
                          double &mid_x, double &mid_y, double &width) const
{
    if (scan.ranges.size() < 3 || scan.angle_increment <= 0.0)
        return false;

    const int first = std::max(
        0, (int)std::ceil((-gap_search_half_angle_ - scan.angle_min) /
                          scan.angle_increment));
    const int last = std::min(
        (int)scan.ranges.size() - 1,
        (int)std::floor((gap_search_half_angle_ - scan.angle_min) /
                        scan.angle_increment));

    if (first >= last)
        return false;

    auto is_near_wall = [&](int index)
    {
        if (index < first || index > last)
            return false;
        const float range = scan.ranges[index];
        return std::isfinite(range) &&
               range >= scan.range_min &&
               range <= scan.range_max &&
               range <= gap_near_max_range_;
    };

    auto collect_wall_segment = [&](int edge, int direction,
                                    std::vector<point_2d> &segment)
    {
        const int max_points = 80;
        const double max_neighbor_gap = 0.10;

        for (int offset = 0; offset < max_points; ++offset)
        {
            const int point_index = edge + direction * offset;
            if (!is_near_wall(point_index))
                break;

            const double angle =
                scan.angle_min + point_index * scan.angle_increment;
            const double range = scan.ranges[point_index];
            point_2d point;
            point.x = range * std::cos(angle);
            point.y = range * std::sin(angle);

            if (!segment.empty() &&
                std::hypot(point.x - segment.back().x,
                           point.y - segment.back().y) > max_neighbor_gap)
            {
                break;
            }
            segment.push_back(point);
        }
    };

    auto fit_wall = [&](const std::vector<point_2d> &segment,
                        double &slope, double &intercept, double &residual)
    {
        if ((int)segment.size() < gap_wall_min_points_)
            return false;

        const double segment_length =
            std::hypot(segment.front().x - segment.back().x,
                       segment.front().y - segment.back().y);
        if (segment_length < gap_wall_min_length_)
            return false;

        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_xy = 0.0;
        double sum_y2 = 0.0;
        for (const point_2d &point : segment)
        {
            sum_x += point.x;
            sum_y += point.y;
            sum_xy += point.x * point.y;
            sum_y2 += point.y * point.y;
        }

        const double count = segment.size();
        const double denominator = count * sum_y2 - sum_y * sum_y;
        if (std::fabs(denominator) < 1e-6)
            return false;

        // 围墙横在车头前，因此用 x = slope * y + intercept 拟合。
        slope = (count * sum_xy - sum_x * sum_y) / denominator;
        intercept = (sum_x - slope * sum_y) / count;

        double squared_error = 0.0;
        for (const point_2d &point : segment)
        {
            const double error = point.x - (slope * point.y + intercept);
            squared_error += error * error;
        }
        residual = std::sqrt(squared_error / count);

        return residual <= gap_wall_max_residual_ &&
               std::fabs(slope) <= 0.30;
    };

    bool found = false;
    double best_score = std::numeric_limits<double>::infinity();
    const double expected_width = (gap_min_width_ + gap_max_width_) / 2.0;

    int index = first + 1;
    while (index < last)
    {
        if (is_near_wall(index))
        {
            ++index;
            continue;
        }

        const int right_edge = index - 1;
        while (index <= last && !is_near_wall(index))
            ++index;
        const int left_edge = index;

        if (right_edge < first || left_edge > last)
            continue;

        std::vector<point_2d> right_wall;
        std::vector<point_2d> left_wall;
        collect_wall_segment(right_edge, -1, right_wall);
        collect_wall_segment(left_edge, 1, left_wall);

        double right_slope = 0.0;
        double right_intercept = 0.0;
        double right_residual = 0.0;
        double left_slope = 0.0;
        double left_intercept = 0.0;
        double left_residual = 0.0;
        if (!fit_wall(right_wall, right_slope, right_intercept, right_residual) ||
            !fit_wall(left_wall, left_slope, left_intercept, left_residual))
            continue;

        if (std::fabs(right_slope - left_slope) > 0.15 ||
            std::fabs(right_intercept - left_intercept) > gap_wall_max_line_offset_)
        {
            continue;
        }

        // 用拟合墙面修正两个内侧端点的 x，y 保留边缘扫描点。
        const double right_y = right_wall.front().y;
        const double left_y = left_wall.front().y;
        const double right_x = right_slope * right_y + right_intercept;
        const double left_x = left_slope * left_y + left_intercept;
        const double candidate_width =
            std::hypot(left_x - right_x, left_y - right_y);
        const double candidate_mid_x = (left_x + right_x) / 2.0;
        const double candidate_mid_y = (left_y + right_y) / 2.0;

        const double lidar_cos = std::cos(lidar_yaw_);
        const double lidar_sin = std::sin(lidar_yaw_);
        const double candidate_base_x =
            lidar_offset_x_ + lidar_cos * candidate_mid_x - lidar_sin * candidate_mid_y;
        const double candidate_base_y =
            lidar_offset_y_ + lidar_sin * candidate_mid_x + lidar_cos * candidate_mid_y;

        if (candidate_width < gap_min_width_ ||
            candidate_width > gap_max_width_ ||
            candidate_base_x < gap_min_forward_offset_ ||
            candidate_base_x > gap_max_forward_offset_ ||
            std::fabs(candidate_base_y) > gap_max_lateral_offset_)
        {
            continue;
        }

        const double score =
            std::fabs(candidate_width - expected_width) +
            0.4 * std::fabs(candidate_base_y) +
            std::fabs(right_intercept - left_intercept) +
            right_residual + left_residual;

        if (score < best_score)
        {
            best_score = score;
            mid_x = candidate_mid_x;
            mid_y = candidate_mid_y;
            width = candidate_width;
            found = true;
        }
    }

    return found;
}

void OURSWITCH::ScanCallback(const sensor_msgs::LaserScan::ConstPtr &msg)
{
    {
        std::lock_guard<std::mutex> lock(gap_mutex_);
        if (!collect_gap_samples_)
            return;
    }

    double mid_x = 0.0;
    double mid_y = 0.0;
    double width = 0.0;
    if (!detectGap(*msg, mid_x, mid_y, width))
        return;

    std::lock_guard<std::mutex> lock(gap_mutex_);
    if (!collect_gap_samples_)
        return;

    gap_mid_x_samples_.push_back(mid_x);
    gap_mid_y_samples_.push_back(mid_y);
    gap_width_samples_.push_back(width);
}

void OURSWITCH::SignalClassCallback(const std_msgs::Int32::ConstPtr &msg)
{
    if (target_locked_)
        return;
    current_signal_class_ = msg->data;
    last_signal_class_time_ = ros::Time::now();
}

void OURSWITCH::SignalDetectionCallback(const std_msgs::Float32MultiArray::ConstPtr &msg)
{
    if (target_locked_)
        return;
    if (msg->data.size() < 4)
        return;

    signal_center_x_ = msg->data[0];
    signal_center_y_ = msg->data[1];
    signal_box_x_l_ = msg->data[2];
    signal_box_x_r_ = msg->data[3];
    last_signal_detection_time_ = ros::Time::now();
    nh_.setParam("center_x", signal_center_x_);
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
    if (INPUT > MAX)
        return MAX;
    else if (INPUT < MIN)
        return MIN;
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
            if (std::abs(error) < 0.05)
                escape_state = 2;
            break;

        case 2:
            error = distance_you_y - safe_R;
            cmd.linear.y = -Limit_Value(Kp_dist * error, max_vel, -max_vel);
            cmd.linear.x = 0;
            if (std::abs(error) < 0.05)
                escape_state = 3;
            break;

        case 3:
            error = distance_qian_x - safe_F;
            cmd.linear.x = Limit_Value(Kp_dist * error, max_vel, -max_vel);
            cmd.linear.y = 0;
            if (std::abs(error) < 0.05)
                escape_state = 4;
            break;

        case 4:
            error = distance_zuo_y - safe_L;
            cmd.linear.y = Limit_Value(Kp_dist * error, max_vel, -max_vel);
            cmd.linear.x = 0;
            if (std::abs(error) < 0.05)
                escape_state = 5;
            break;

        case 5:
            error = distance_qian_x - safe_F;
            cmd.linear.x = Limit_Value(Kp_dist * error, max_vel, -max_vel);
            cmd.linear.y = 0;
            if (std::abs(error) < 0.05)
                escape_state = 6;
            break;

        case 6:
            error = distance_you_y - safe_R;
            cmd.linear.y = -Limit_Value(Kp_dist * error, max_vel, -max_vel);
            cmd.linear.x = 0;
            if (std::abs(error) < 0.05)
                escape_state = 7;
            break;

        case 7:
            error = distance_hou_x - safe_B;
            cmd.linear.x = -Limit_Value(Kp_dist * error, max_vel, -max_vel);
            cmd.linear.y = 0;
            if (std::abs(error) < 0.05)
                escape_state = 8;
            break;

        case 8:
            error = distance_zuo_y - safe_L;
            cmd.linear.y = Limit_Value(Kp_dist * error, max_vel, -max_vel);
            cmd.linear.x = 0;
            if (std::abs(error) < 0.05)
                escape_state = 9;
            break;

        case 9:
            error = distance_hou_x - safe_B;
            cmd.linear.x = -Limit_Value(Kp_dist * error, max_vel, -max_vel);
            cmd.linear.y = 0;
            if (std::abs(error) < 0.05)
                escape_state = 10;
            break;

        case 10:
            error = distance_you_y - safe_R2;
            cmd.linear.y = -Limit_Value(Kp_dist * error, max_vel, -max_vel);
            cmd.linear.x = 0;
            if (std::abs(error) < 0.05)
                escape_state = 11;
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
// 物品领取区交接 (快速转动 + 静止扫码)
// =========================================================================
void OURSWITCH::GotoB()
{
    ROS_INFO("Entering GotoB state: Multi-point QR search");

    struct Pose
    {
        double x;
        double y;
        double yaw;
    };

    std::vector<Pose> qr_points = {
        {-1.56, -0.50, 3.14},
        {-1.56, -0.70, -1.57},
        {-1.56, -0.30, 1.57},
        {-1.56, -0.50, 0}};

    int scan_done = 0;
    bool start_qr_scan_sent = false;

    const int view_count = 6;
    const double turn_speed = 1.0;
    const double turn_duration = (2.0 * M_PI / view_count) / turn_speed; // 时间乘2防止转速上限
    const double settle_duration = 0.4;
    const double scan_duration = 1.0;
    const double nav_timeout = 15.0;

    for (int i = 0; i < (int)qr_points.size() && ros::ok(); ++i)
    {
        nh_.getParam("qr_scan_done", scan_done);
        if (scan_done == 1)
        {
            ROS_INFO("All 3 QR codes already found before point %d", i + 1);
            break;
        }

        ROS_INFO("Navigating to QR observation point %d", i + 1);
        sendPos(qr_points[i].x, qr_points[i].y, qr_points[i].yaw);

        bool arrived = ac_.waitForResult(ros::Duration(nav_timeout));
        if (!arrived)
        {
            ROS_WARN("QR point %d timeout, cancel and go next", i + 1);
            ac_.cancelGoal();
            continue;
        }

        if (ac_.getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
        {
            ROS_WARN("QR point %d unreachable, state=%s",
                     i + 1, ac_.getState().toString().c_str());
            continue;
        }

        ROS_INFO("Arrived QR point %d", i + 1);

        geometry_msgs::Twist stop_cmd;
        cmd_vel_pub__.publish(stop_cmd);
        ros::Duration(0.3).sleep();

        // 关键：到第一个成功抵达的观察点后，再通知 AI.py 开始最终二维码流程
        if (!start_qr_scan_sent)
        {
            nh_.setParam("start_qr_scan", 1);
            start_qr_scan_sent = true;
            ROS_INFO("start_qr_scan set to 1 after arriving first QR point");
        }

        nh_.getParam("qr_scan_done", scan_done);
        if (scan_done == 1)
        {
            ROS_INFO("All 3 QR codes found before static scanning at point %d", i + 1);
            break;
        }

        ros::Rate rate(20);

        for (int view = 0; view <= view_count && ros::ok(); ++view)
        {
            nh_.getParam("qr_scan_done", scan_done);
            if (scan_done == 1)
            {
                ROS_INFO("All 3 QR codes found at point %d", i + 1);
                break;
            }

            if (view > 0)
            {
                geometry_msgs::Twist turn_cmd;
                turn_cmd.angular.z = turn_speed;

                ros::Time turn_start = ros::Time::now();
                while (ros::ok() &&
                       (ros::Time::now() - turn_start).toSec() < turn_duration)
                {
                    cmd_vel_pub__.publish(turn_cmd);
                    ros::spinOnce();
                    rate.sleep();
                }
            }

            cmd_vel_pub__.publish(stop_cmd);
            ros::Duration(settle_duration).sleep();

            ROS_INFO("QR point %d, static view %d/%d",
                     i + 1, view + 1, view_count);

            ros::Time scan_start = ros::Time::now();
            while (ros::ok() &&
                   (ros::Time::now() - scan_start).toSec() < scan_duration)
            {
                nh_.getParam("qr_scan_done", scan_done);
                if (scan_done == 1)
                {
                    ROS_INFO("All 3 QR codes found at static view %d", view + 1);
                    break;
                }

                ros::spinOnce();
                rate.sleep();
            }

            if (scan_done == 1)
            {
                break;
            }
        }

        cmd_vel_pub__.publish(stop_cmd);

        nh_.getParam("qr_scan_done", scan_done);
        if (scan_done == 1)
        {
            ROS_INFO("QR search completed at point %d", i + 1);
            break;
        }

        ROS_WARN("QR codes not complete after %d static views at point %d, go next point",
                 view_count, i + 1);
    }

    nh_.getParam("qr_scan_done", scan_done);

    if (!start_qr_scan_sent)
    {
        ROS_WARN("No QR observation point reached. Force start_qr_scan=1 to avoid AI waiting forever.");
        nh_.setParam("start_qr_scan", 1);
        start_qr_scan_sent = true;
    }

    if (scan_done == 0)
    {
        ROS_WARN("QR codes still incomplete after all observation points. Starting fallback stop-and-look search.");

        geometry_msgs::Twist stop_cmd;
        geometry_msgs::Twist turn_cmd;
        turn_cmd.angular.z = turn_speed;
        ros::Rate rate(20);

        while (ros::ok())
        {
            cmd_vel_pub__.publish(stop_cmd);
            ros::Duration(settle_duration).sleep();

            ros::Time scan_start = ros::Time::now();
            while (ros::ok() &&
                   (ros::Time::now() - scan_start).toSec() < scan_duration)
            {
                nh_.getParam("qr_scan_done", scan_done);
                if (scan_done == 1)
                {
                    break;
                }

                ros::spinOnce();
                rate.sleep();
            }

            if (scan_done == 1)
            {
                break;
            }

            ros::Time turn_start = ros::Time::now();
            while (ros::ok() &&
                   (ros::Time::now() - turn_start).toSec() < turn_duration)
            {
                cmd_vel_pub__.publish(turn_cmd);
                ros::spinOnce();
                rate.sleep();
            }

            nh_.getParam("qr_scan_done", scan_done);
            if (scan_done == 1)
            {
                break;
            }
        }

        cmd_vel_pub__.publish(stop_cmd);
    }

    ROS_INFO("Waiting for AI.py to finish LLM matching and broadcasting...");

    int task1_done = 0;
    ros::Rate wait_rate(10);

    while (task1_done == 0 && ros::ok())
    {
        nh_.getParam("task1_all_done", task1_done);
        ros::spinOnce();
        wait_rate.sleep();
    }

    ROS_INFO("AI.py task1 completed. Switching to XingHuoAI state.");
    current_state = XingHuoAI_;
}

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

// void OURSWITCH::GotoC(int target_num)
// {
//     // 使用现有的 distance_qian_x 调整小车与前方目标之间的距离。
//     auto adjustFrontDistance = [this]() -> bool
//     {
//         const double target_distance = 0.20;
//         const double distance_tolerance = 0.02;
//         const double stable_duration = 0.40;
//         const double adjustment_timeout = 10.0;
//         const double maximum_linear_speed = 0.15;
//         const double maximum_angular_speed = 0.40;

//         PID front_pid = {};
//         front_pid.kp =
//             nh_.param("warehouse_front_kp", Kp_dist);
//         front_pid.ki =
//             nh_.param("warehouse_front_ki", 0.0);
//         front_pid.kd =
//             nh_.param("warehouse_front_kd", 0.0);

//         // 清除构造函数中的默认值，必须等待 /ultra 回调提供新数据。
//         distance_qian_x =
//             std::numeric_limits<double>::quiet_NaN();

//         // 保持进入微调阶段时的车头方向。
//         const double hold_yaw = yaw;

//         ros::Time adjustment_start = ros::Time::now();
//         ros::Time stable_start;
//         ros::Rate pid_rate(20);

//         bool adjustment_succeeded = false;

//         ROS_INFO(
//             "Starting distance_qian_x PID: "
//             "target=%.3f tolerance=%.3f timeout=%.1f",
//             target_distance,
//             distance_tolerance,
//             adjustment_timeout);

//         while (ros::ok() &&
//                (ros::Time::now() - adjustment_start).toSec() <
//                    adjustment_timeout)
//         {
//             geometry_msgs::Twist pid_cmd;

//             const double measured_distance =
//                 distance_qian_x;

//             // 等待 /ultra 提供有效的新数据。
//             if (!std::isfinite(measured_distance) ||
//                 measured_distance <= 0.02 ||
//                 measured_distance > 5.0)
//             {
//                 stable_start = ros::Time(0);
//                 cmd_vel_pub__.publish(pid_cmd);

//                 ROS_WARN_THROTTLE(
//                     1.0,
//                     "Waiting for valid distance_qian_x");

//                 pid_rate.sleep();
//                 continue;
//             }

//             const double distance_error =
//                 measured_distance - target_distance;

//             if (std::fabs(distance_error) <=
//                 distance_tolerance)
//             {
//                 // 距离已经满足要求，停止前后移动。
//                 pid_cmd.linear.x = 0.0;

//                 if (stable_start.isZero())
//                 {
//                     stable_start = ros::Time::now();
//                 }

//                 if ((ros::Time::now() -
//                      stable_start)
//                         .toSec() >=
//                     stable_duration)
//                 {
//                     adjustment_succeeded = true;
//                     cmd_vel_pub__.publish(pid_cmd);
//                     break;
//                 }
//             }
//             else
//             {
//                 stable_start = ros::Time(0);

//                 // distance_qian_x > 0.2：向前移动。
//                 // distance_qian_x < 0.2：向后移动。
//                 pid_cmd.linear.x =
//                     PID_Realize2(
//                         &front_pid,
//                         distance_error,
//                         maximum_linear_speed,
//                         -maximum_linear_speed,
//                         0.5);
//             }

//             pid_cmd.linear.y = 0.0;

//             // 保持进入 PID 阶段时的车头方向。
//             const double yaw_error =
//                 std::atan2(
//                     std::sin(hold_yaw - yaw),
//                     std::cos(hold_yaw - yaw));

//             pid_cmd.angular.z =
//                 Limit_Value(
//                     Kp_yaw * yaw_error,
//                     maximum_angular_speed,
//                     -maximum_angular_speed);

//             cmd_vel_pub__.publish(pid_cmd);

//             ROS_INFO_THROTTLE(
//                 0.5,
//                 "Front PID: distance_qian_x=%.3f "
//                 "error=%.3f linear=%.3f angular=%.3f",
//                 measured_distance,
//                 distance_error,
//                 pid_cmd.linear.x,
//                 pid_cmd.angular.z);

//             pid_rate.sleep();
//         }

//         // 无论成功还是超时，都强制停车。
//         geometry_msgs::Twist stop_cmd;

//         for (int i = 0; i < 3; ++i)
//         {
//             cmd_vel_pub__.publish(stop_cmd);
//             ros::Duration(0.05).sleep();
//         }

//         if (adjustment_succeeded)
//         {
//             ROS_INFO(
//                 "distance_qian_x PID completed: "
//                 "front distance is stable at %.2f m",
//                 target_distance);
//         }
//         else
//         {
//             ROS_WARN(
//                 "distance_qian_x PID timed out; "
//                 "vehicle stopped");
//         }

//         return adjustment_succeeded;
//     };

//     std::string target_warehouse = "UNKNOWN";
//     target_locked_ = false;

//     if (target_num == 1)
//     {
//         ROS_INFO(
//             "Entering GotoC1 state: "
//             "Real Car Warehouse Matching");

//         nh_.getParam(
//             "real_class",
//             target_warehouse);
//     }
//     else
//     {
//         ROS_INFO(
//             "Entering GotoC2 state: "
//             "Sim Car Warehouse Matching");

//         nh_.getParam(
//             "sim_class",
//             target_warehouse);
//     }

//     nh_.setParam(
//         "auto_park_target",
//         target_warehouse);
//     nh_.setParam(
//         "start_auto_park",
//         0);
//     nh_.setParam(
//         "auto_park_status",
//         "IDLE");

//     int target_class = -1;

//     if (target_warehouse.find("食品") !=
//         std::string::npos)
//     {
//         target_class = 0;
//     }
//     else if (
//         target_warehouse.find("日用品") !=
//             std::string::npos ||
//         target_warehouse.find("用品") !=
//             std::string::npos)
//     {
//         target_class = 1;
//     }
//     else if (
//         target_warehouse.find("电子") !=
//             std::string::npos ||
//         target_warehouse.find("电") !=
//             std::string::npos ||
//         target_warehouse.find("生产") !=
//             std::string::npos)
//     {
//         target_class = 2;
//     }

//     if (target_class < 0)
//     {
//         ROS_ERROR(
//             "Unknown target warehouse: %s",
//             target_warehouse.c_str());
//     }

//     struct Pose
//     {
//         double x;
//         double y;
//         double yaw;
//     };

//     std::vector<Pose> search_points = {
//         {-1.3, -2.4, 1.57},
//         {0.6, -2.3, 1.57},
//         {2.0, -2.3, 1.57}};

//     // 已取得有效目标坐标，不代表导航一定成功。
//     bool target_found = false;

//     // 只有该变量为 true 时才允许进行语音播报。
//     bool ready_to_announce = false;

//     double target_dx = 0.0;
//     double target_dy = 0.0;
//     double target_line_a = 0.0;

//     for (int i = 0;
//          i < static_cast<int>(search_points.size()) &&
//          ros::ok();
//          ++i)
//     {
//         ROS_DEBUG(
//             "Navigating to observation point %d",
//             i + 1);

//         current_signal_class_ = -1;
//         last_signal_class_time_ = ros::Time(0);
//         last_signal_detection_time_ = ros::Time(0);

//         sendPos(
//             search_points[i].x,
//             search_points[i].y,
//             search_points[i].yaw);

//         bool arrived =
//             ac_.waitForResult(
//                 ros::Duration(20.0));

//         if (!arrived)
//         {
//             ROS_WARN(
//                 "Point %d timeout, skip",
//                 i + 1);

//             ac_.cancelGoal();
//             ac_.waitForResult(
//                 ros::Duration(0.5));
//             continue;
//         }

//         if (ac_.getState() !=
//             actionlib::SimpleClientGoalState::SUCCEEDED)
//         {
//             ROS_WARN(
//                 "Point %d unreachable, state=%s",
//                 i + 1,
//                 ac_.getState().toString().c_str());
//             continue;
//         }

//         ROS_DEBUG(
//             "Arrived point %d, "
//             "starting stop-and-look signal search",
//             i + 1);

//         const int view_count = 6;
//         const double turn_speed = 1.0;
//         const double turn_duration =
//             (2.0 * M_PI / view_count) /
//             turn_speed;
//         const double settle_duration = 0.4;
//         const double recognition_duration = 1.0;

//         bool leave_observation_point = false;
//         ros::Rate rate(20);

//         for (int view = 0;
//              view < view_count && ros::ok();
//              ++view)
//         {
//             if (view > 0)
//             {
//                 target_locked_ = true;

//                 geometry_msgs::Twist turn_cmd;
//                 turn_cmd.angular.z = turn_speed;

//                 ros::Time turn_start =
//                     ros::Time::now();

//                 while (ros::ok() &&
//                        (ros::Time::now() -
//                         turn_start)
//                                .toSec() <
//                            turn_duration)
//                 {
//                     cmd_vel_pub__.publish(turn_cmd);
//                     ros::spinOnce();
//                     rate.sleep();
//                 }
//             }

//             geometry_msgs::Twist stop_cmd;
//             cmd_vel_pub__.publish(stop_cmd);
//             ros::Duration(settle_duration).sleep();

//             current_signal_class_ = -1;
//             last_signal_class_time_ = ros::Time(0);
//             last_signal_detection_time_ = ros::Time(0);
//             target_locked_ = false;

//             ROS_DEBUG(
//                 "Observation point %d, "
//                 "static view %d/%d",
//                 i + 1,
//                 view + 1,
//                 view_count);

//             ros::Time recognition_start =
//                 ros::Time::now();

//             while (ros::ok() &&
//                    (ros::Time::now() -
//                     recognition_start)
//                            .toSec() <
//                        recognition_duration)
//             {
//                 ros::spinOnce();

//                 bool class_recent =
//                     !last_signal_class_time_.isZero() &&
//                     (ros::Time::now() -
//                      last_signal_class_time_)
//                             .toSec() <
//                         1.0;

//                 bool detection_recent =
//                     !last_signal_detection_time_.isZero() &&
//                     (ros::Time::now() -
//                      last_signal_detection_time_)
//                             .toSec() <
//                         1.0;

//                 if (class_recent &&
//                     detection_recent &&
//                     current_signal_class_ == target_class)
//                 {
//                     ROS_DEBUG(
//                         "Target class matched. "
//                         "class=%d center_x=%.1f",
//                         current_signal_class_,
//                         signal_center_x_);

//                     geometry_msgs::Twist detection_stop_cmd;
//                     cmd_vel_pub__.publish(
//                         detection_stop_cmd);

//                     ros::Duration(0.3).sleep();

//                     bool frozen = false;
//                     ros::Time freeze_start =
//                         ros::Time::now();

//                     while (ros::ok() &&
//                            (ros::Time::now() -
//                             freeze_start)
//                                    .toSec() <
//                                1.0)
//                     {
//                         ros::spinOnce();

//                         bool class_ok =
//                             !last_signal_class_time_.isZero() &&
//                             (ros::Time::now() -
//                              last_signal_class_time_)
//                                     .toSec() <
//                                 1.0;

//                         bool detection_ok =
//                             !last_signal_detection_time_.isZero() &&
//                             (ros::Time::now() -
//                              last_signal_detection_time_)
//                                     .toSec() <
//                                 1.0;

//                         if (class_ok &&
//                             detection_ok &&
//                             current_signal_class_ ==
//                                 target_class)
//                         {
//                             locked_signal_class_ =
//                                 current_signal_class_;
//                             locked_center_x_ =
//                                 signal_center_x_;
//                             locked_box_x_l_ =
//                                 signal_box_x_l_;
//                             locked_box_x_r_ =
//                                 signal_box_x_r_;

//                             target_locked_ = true;
//                             frozen = true;

//                             ROS_DEBUG(
//                                 "Target locked after stop. "
//                                 "class=%d center=%.1f "
//                                 "left=%.1f right=%.1f",
//                                 locked_signal_class_,
//                                 locked_center_x_,
//                                 locked_box_x_l_,
//                                 locked_box_x_r_);
//                             break;
//                         }

//                         ros::Duration(0.05).sleep();
//                     }

//                     if (!frozen)
//                     {
//                         ROS_WARN(
//                             "Target matched, but no stable "
//                             "post-stop detection found");
//                         continue;
//                     }

//                     ourgoal::getLaserPoint srv;

//                     srv.request.center_x =
//                         std::max(
//                             0,
//                             std::min(
//                                 639,
//                                 static_cast<int>(
//                                     std::round(
//                                         locked_center_x_))));

//                     srv.request.left_x =
//                         std::max(
//                             0,
//                             std::min(
//                                 639,
//                                 static_cast<int>(
//                                     std::round(
//                                         locked_box_x_l_))));

//                     srv.request.right_x =
//                         std::max(
//                             0,
//                             std::min(
//                                 639,
//                                 static_cast<int>(
//                                     std::round(
//                                         locked_box_x_r_))));

//                     srv.request.mode = false;

//                     if (srv.request.left_x <
//                             srv.request.right_x &&
//                         vision_gettool_client_.call(srv))
//                     {
//                         target_dx =
//                             (srv.response.dx_left +
//                              srv.response.dx_right) /
//                             2.0;

//                         target_dy =
//                             (srv.response.dy_left +
//                              srv.response.dy_right) /
//                             2.0;

//                         target_line_a =
//                             srv.response.line_a;

//                         ROS_INFO(
//                             "Signal metric center: "
//                             "dx=%.3f dy=%.3f line_a=%.3f",
//                             target_dx,
//                             target_dy,
//                             target_line_a);

//                         nh_.setParam(
//                             "signal_target_dx",
//                             target_dx);
//                         nh_.setParam(
//                             "signal_target_dy",
//                             target_dy);
//                         nh_.setParam(
//                             "signal_target_line_a",
//                             target_line_a);

//                         point_2d target_in_base;
//                         target_in_base.x = target_dx;
//                         target_in_base.y = target_dy;

//                         double k = target_line_a;

//                         double wall_tangent_yaw =
//                             (k == 255)
//                                 ? M_PI / 2.0
//                                 : std::atan(k);

//                         if (wall_tangent_yaw < 0.0)
//                         {
//                             wall_tangent_yaw += M_PI;
//                         }

//                         double approach_yaw_in_base =
//                             wall_tangent_yaw -
//                             M_PI / 2.0;

//                         double car_x =
//                             nh_.param("CarX", 0.0);
//                         double car_y =
//                             nh_.param("CarY", 0.0);
//                         double car_yaw =
//                             nh_.param("CarYaw", 0.0);

//                         point_2d target_in_map =
//                             rotate(
//                                 target_in_base,
//                                 car_yaw);

//                         target_in_map =
//                             translate(
//                                 target_in_map,
//                                 car_x,
//                                 car_y);

//                         double target_yaw =
//                             approach_yaw_in_base +
//                             car_yaw;

//                         const double stop_distance = 0.30;

//                         point_2d map_target;

//                         map_target.x =
//                             target_in_map.x -
//                             stop_distance *
//                                 std::cos(target_yaw);

//                         map_target.y =
//                             target_in_map.y -
//                             stop_distance *
//                                 std::sin(target_yaw);

//                         const double original_goal_x =
//                             map_target.x;
//                         const double original_goal_y =
//                             map_target.y;

//                         const double min_goal_x = -1.9;
//                         const double max_goal_x = 2.4;
//                         const double min_goal_y = -2.9;
//                         const double max_goal_y = -1.0;

//                         if (map_target.x < min_goal_x)
//                         {
//                             map_target.x = min_goal_x;
//                         }
//                         else if (map_target.x > max_goal_x)
//                         {
//                             map_target.x = max_goal_x;
//                         }

//                         if (map_target.y < min_goal_y)
//                         {
//                             map_target.y = min_goal_y;
//                         }
//                         else if (map_target.y > max_goal_y)
//                         {
//                             map_target.y = max_goal_y;
//                         }

//                         const bool goal_clamped =
//                             map_target.x != original_goal_x ||
//                             map_target.y != original_goal_y;

//                         const double face_target_dx =
//                             target_in_map.x -
//                             map_target.x;

//                         const double face_target_dy =
//                             target_in_map.y -
//                             map_target.y;

//                         const double actual_stop_distance =
//                             std::hypot(
//                                 face_target_dx,
//                                 face_target_dy);

//                         if (actual_stop_distance > 1e-6)
//                         {
//                             target_yaw =
//                                 std::atan2(
//                                     face_target_dy,
//                                     face_target_dx);
//                         }

//                         ROS_INFO(
//                             "Warehouse target in map: "
//                             "x=%.3f y=%.3f",
//                             target_in_map.x,
//                             target_in_map.y);

//                         if (goal_clamped)
//                         {
//                             ROS_WARN(
//                                 "Parking goal clamped: "
//                                 "(%.3f, %.3f) -> "
//                                 "(%.3f, %.3f)",
//                                 original_goal_x,
//                                 original_goal_y,
//                                 map_target.x,
//                                 map_target.y);
//                         }

//                         ROS_INFO(
//                             "Parking goal in map: "
//                             "x=%.3f y=%.3f yaw=%.3f "
//                             "distance_to_target=%.3f",
//                             map_target.x,
//                             map_target.y,
//                             target_yaw,
//                             actual_stop_distance);

//                         // 已成功获得目标坐标。
//                         target_found = true;

//                         sendPos(
//                             map_target.x,
//                             map_target.y,
//                             target_yaw);

//                         bool park_finished =
//                             ac_.waitForResult(
//                                 ros::Duration(15.0));

//                         actionlib::SimpleClientGoalState
//                             park_state = ac_.getState();

//                         if (park_finished &&
//                             park_state ==
//                                 actionlib::
//                                     SimpleClientGoalState::
//                                         SUCCEEDED)
//                         {
//                             ROS_INFO(
//                                 "Arrived at warehouse "
//                                 "navigation pose");

//                             // 到达导航点以后，使用
//                             // distance_qian_x 调整到 0.20 米。
//                             bool front_adjusted =
//                                 adjustFrontDistance();

//                             if (front_adjusted)
//                             {
//                                 nh_.setParam(
//                                     "auto_park_status",
//                                     "DONE");

//                                 ready_to_announce = true;
//                             }
//                             else
//                             {
//                                 nh_.setParam(
//                                     "auto_park_status",
//                                     "FAILED");

//                                 ready_to_announce = false;

//                                 ROS_WARN(
//                                     "Reached warehouse pose, "
//                                     "but distance_qian_x PID "
//                                     "did not reach 0.20 m; "
//                                     "skip announcement");
//                             }
//                         }
//                         else
//                         {
//                             if (!park_finished)
//                             {
//                                 ac_.cancelGoal();
//                                 ac_.waitForResult(
//                                     ros::Duration(0.5));
//                             }

//                             ROS_WARN(
//                                 "Warehouse target was found, "
//                                 "but parking failed: "
//                                 "finished=%s state=%s. "
//                                 "Stop searching and continue.",
//                                 park_finished
//                                     ? "true"
//                                     : "false",
//                                 park_state
//                                     .toString()
//                                     .c_str());

//                             nh_.setParam(
//                                 "auto_park_status",
//                                 "FAILED");

//                             // 保留之前约定：
//                             // 找到目标但导航无法到达时，
//                             // 直接播报并进入下一阶段。
//                             ready_to_announce = true;
//                         }

//                         target_locked_ = false;
//                         leave_observation_point = true;
//                         break;
//                     }
//                     else
//                     {
//                         ROS_WARN(
//                             "Target class matched, but "
//                             "/srv_getLaserPoint failed");

//                         target_locked_ = false;
//                     }
//                 }

//                 rate.sleep();
//             }

//             if (target_found ||
//                 leave_observation_point)
//             {
//                 break;
//             }
//         }

//         geometry_msgs::Twist final_stop_cmd;
//         cmd_vel_pub__.publish(final_stop_cmd);

//         if (target_found)
//         {
//             break;
//         }

//         ROS_WARN(
//             "Target not found at point %d, go next",
//             i + 1);
//     }

//     // move_base 成功时，只有距离 PID 完成后才播报。
//     // move_base 无法到达时，按照之前要求直接播报。
//     if (target_found &&
//         ready_to_announce &&
//         target_num == 1)
//     {
//         std::string item;
//         std::string room;

//         nh_.getParam("real_item", item);
//         nh_.getParam("real_room", room);

//         char tts_cmd[512];

//         std::snprintf(
//             tts_cmd,
//             sizeof(tts_cmd),
//             "espeak -v zh+f2 "
//             "\"已将%s放入%s\" -s 130",
//             item.c_str(),
//             room.c_str());

//         system(tts_cmd);
//     }

//     if (!target_found)
//     {
//         ROS_ERROR(
//             "Failed to find target warehouse "
//             "after all observation points");

//         nh_.setParam(
//             "auto_park_status",
//             "FAILED");
//     }

//     current_state =
//         (target_num == 1)
//             ? GOTOC2_
//             : Gazebo_;
// }

void OURSWITCH::GotoC(int target_num)
{
    auto publishStop = [this]()
    {
        geometry_msgs::Twist stop_cmd;

        for (int i = 0; i < 3; ++i)
        {
            cmd_vel_pub__.publish(stop_cmd);
            ros::WallDuration(0.05).sleep();
        }
    };

    // 到达导航停车点后，根据 RKNN 识别框中心进行横向对正。
    auto adjustLateralPosition =
        [this, &publishStop](
            bool require_heading_check,
            double target_map_yaw,
            int expected_class) -> bool
    {
        const double target_center_x =
            nh_.param("warehouse_target_center_x", 320.0);
        const double center_tolerance =
            nh_.param("warehouse_center_tolerance_px", 15.0);
        const double lateral_kp =
            nh_.param("warehouse_lateral_kp", 0.0005);
        const double maximum_lateral_speed =
            nh_.param("warehouse_max_lateral_speed", 0.05);
        const double adjustment_timeout =
            nh_.param("warehouse_lateral_timeout", 5.0);
        const double maximum_detection_age = 1.0;
        const double maximum_angular_speed = 0.20;
        const double maximum_heading_error =
            20.0 * M_PI / 180.0;
        const int required_stable_frames = 5;

        if (expected_class < 0 ||
            target_center_x < 0.0 ||
            target_center_x > 640.0)
        {
            ROS_WARN(
                "Cannot run RKNN lateral alignment: "
                "class=%d target_center_x=%.1f",
                expected_class,
                target_center_x);

            publishStop();
            return false;
        }

        if (require_heading_check)
        {
            double current_map_yaw = 0.0;

            if (!nh_.getParam("CarYaw", current_map_yaw) ||
                !std::isfinite(current_map_yaw))
            {
                ROS_WARN(
                    "Cannot run RKNN lateral alignment after "
                    "navigation failure: CarYaw is unavailable");

                publishStop();
                return false;
            }

            const double map_heading_error =
                std::atan2(
                    std::sin(target_map_yaw - current_map_yaw),
                    std::cos(target_map_yaw - current_map_yaw));

            if (std::fabs(map_heading_error) >
                maximum_heading_error)
            {
                ROS_WARN(
                    "Cannot run RKNN lateral alignment safely: "
                    "heading error %.1f deg exceeds %.1f deg",
                    map_heading_error * 180.0 / M_PI,
                    maximum_heading_error * 180.0 / M_PI);

                publishStop();
                return false;
            }
        }

        publishStop();

        // 丢弃导航过程中冻结的结果，只接受停车后的新识别数据。
        current_signal_class_ = -1;
        last_signal_class_time_ = ros::Time(0);
        last_signal_detection_time_ = ros::Time(0);
        target_locked_ = false;

        int stable_frames = 0;
        bool adjustment_succeeded = false;
        ros::Time last_counted_detection_time(0);
        const ros::WallTime adjustment_start =
            ros::WallTime::now();
        ros::WallRate lateral_rate(20.0);

        ROS_INFO(
            "Starting RKNN lateral alignment: "
            "target_center=%.1f tolerance=%.1f "
            "max_speed=%.3f timeout=%.1f",
            target_center_x,
            center_tolerance,
            maximum_lateral_speed,
            adjustment_timeout);

        while (ros::ok() &&
               (ros::WallTime::now() - adjustment_start).toSec() <
                   adjustment_timeout)
        {
            geometry_msgs::Twist lateral_cmd;
            const ros::Time now = ros::Time::now();

            const bool class_recent =
                !last_signal_class_time_.isZero() &&
                (now - last_signal_class_time_).toSec() <
                    maximum_detection_age;

            const bool detection_recent =
                !last_signal_detection_time_.isZero() &&
                (now - last_signal_detection_time_).toSec() <
                    maximum_detection_age;

            const bool target_valid =
                class_recent &&
                detection_recent &&
                current_signal_class_ == expected_class &&
                std::isfinite(signal_center_x_) &&
                signal_center_x_ >= 0.0 &&
                signal_center_x_ <= 640.0;

            if (!target_valid)
            {
                stable_frames = 0;
                cmd_vel_pub__.publish(lateral_cmd);

                ROS_WARN_THROTTLE(
                    1.0,
                    "Waiting for a fresh RKNN detection of "
                    "target class %d during lateral alignment",
                    expected_class);

                lateral_rate.sleep();
                continue;
            }

            const double center_error =
                target_center_x - signal_center_x_;

            const bool new_detection =
                last_signal_detection_time_ !=
                last_counted_detection_time;

            if (new_detection)
            {
                last_counted_detection_time =
                    last_signal_detection_time_;
            }

            if (std::fabs(center_error) <=
                center_tolerance)
            {
                lateral_cmd.linear.y = 0.0;

                if (new_detection)
                {
                    ++stable_frames;
                }
            }
            else
            {
                stable_frames = 0;

                // 图像 x 向右增大，车体 y 向左为正。
                lateral_cmd.linear.y =
                    Limit_Value(
                        lateral_kp * center_error,
                        maximum_lateral_speed,
                        -maximum_lateral_speed);
            }

            double control_yaw = yaw;
            nh_.getParam("CarYaw", control_yaw);

            const double yaw_error =
                std::atan2(
                    std::sin(
                        target_map_yaw - control_yaw),
                    std::cos(
                        target_map_yaw - control_yaw));

            lateral_cmd.angular.z =
                Limit_Value(
                    Kp_yaw * yaw_error,
                    maximum_angular_speed,
                    -maximum_angular_speed);

            cmd_vel_pub__.publish(lateral_cmd);

            ROS_INFO_THROTTLE(
                0.5,
                "RKNN lateral alignment: center=%.1f "
                "error=%.1f linear_y=%.3f stable=%d/%d",
                signal_center_x_,
                center_error,
                lateral_cmd.linear.y,
                stable_frames,
                required_stable_frames);

            if (stable_frames >= required_stable_frames)
            {
                adjustment_succeeded = true;
                break;
            }

            lateral_rate.sleep();
        }

        publishStop();
        target_locked_ = true;

        if (adjustment_succeeded)
        {
            ROS_INFO(
                "RKNN lateral alignment completed: "
                "target centered at %.1f px",
                target_center_x);
        }
        else
        {
            ROS_WARN(
                "RKNN lateral alignment failed or timed out; "
                "vehicle stopped");
        }

        return adjustment_succeeded;
    };

    // 使用现有成员变量 distance_qian_x，将前方距离调整到 0.20 m。
    //
    // require_heading_check:
    //   true  -> move_base 未成功到达，需要先检查当前朝向是否基本朝向目标。
    //   false -> move_base 已成功到达目标姿态，不再额外检查。
    auto adjustFrontDistance =
        [this, &publishStop](
            bool require_heading_check,
            double target_map_yaw) -> bool
    {
        const double target_distance = 0.20;
        const double distance_tolerance = 0.02;
        const double minimum_valid_distance = 0.05;
        const double maximum_valid_distance = 0.80;
        const double maximum_linear_speed = 0.15;
        const double maximum_angular_speed = 0.40;
        const double adjustment_timeout = 5.0;
        const double sample_timeout = 0.50;
        const double maximum_heading_error = 20.0 * M_PI / 180.0;
        const int required_stable_samples = 3;

        if (require_heading_check)
        {
            double current_map_yaw = 0.0;

            if (!nh_.getParam("CarYaw", current_map_yaw) ||
                !std::isfinite(current_map_yaw))
            {
                ROS_WARN(
                    "Cannot run distance PID after navigation failure: "
                    "CarYaw is unavailable");

                publishStop();
                return false;
            }

            const double map_heading_error =
                std::atan2(
                    std::sin(target_map_yaw - current_map_yaw),
                    std::cos(target_map_yaw - current_map_yaw));

            if (std::fabs(map_heading_error) >
                maximum_heading_error)
            {
                ROS_WARN(
                    "Cannot run distance PID safely: "
                    "heading error %.1f deg exceeds %.1f deg",
                    map_heading_error * 180.0 / M_PI,
                    maximum_heading_error * 180.0 / M_PI);

                publishStop();
                return false;
            }
        }

        PID front_pid = {};

        front_pid.kp =
            nh_.param("warehouse_front_kp", Kp_dist);
        front_pid.ki =
            nh_.param("warehouse_front_ki", 0.0);
        front_pid.kd =
            nh_.param("warehouse_front_kd", 0.0);

        int stable_samples = 0;
        bool adjustment_succeeded = false;

        const ros::WallTime adjustment_start =
            ros::WallTime::now();

        ros::WallRate pid_rate(20.0);

        ROS_INFO(
            "Starting distance_qian_x PID: "
            "target=%.3f valid_range=[%.2f, %.2f] "
            "tolerance=%.3f timeout=%.1f",
            target_distance,
            minimum_valid_distance,
            maximum_valid_distance,
            distance_tolerance,
            adjustment_timeout);

        while (ros::ok() &&
               (ros::WallTime::now() -
                adjustment_start)
                       .toSec() <
                   adjustment_timeout)
        {
            // 清除旧值，确保本轮控制等待 /ultra 回调提供新数据。
            distance_qian_x =
                std::numeric_limits<double>::quiet_NaN();

            const ros::WallTime sample_wait_start =
                ros::WallTime::now();

            while (ros::ok() &&
                   !std::isfinite(distance_qian_x) &&
                   (ros::WallTime::now() -
                    sample_wait_start)
                           .toSec() <
                       sample_timeout)
            {
                // 主程序已经启动 AsyncSpinner，
                // /ultra 回调会在后台更新 distance_qian_x。
                ros::WallDuration(0.01).sleep();
            }

            const double measured_distance =
                distance_qian_x;

            geometry_msgs::Twist pid_cmd;

            if (!std::isfinite(measured_distance))
            {
                stable_samples = 0;
                cmd_vel_pub__.publish(pid_cmd);

                ROS_WARN_THROTTLE(
                    1.0,
                    "No fresh distance_qian_x received");

                pid_rate.sleep();
                continue;
            }

            if (measured_distance <
                    minimum_valid_distance ||
                measured_distance >
                    maximum_valid_distance)
            {
                stable_samples = 0;
                cmd_vel_pub__.publish(pid_cmd);

                ROS_WARN_THROTTLE(
                    1.0,
                    "Reject unsafe distance_qian_x=%.3f; "
                    "valid range is [%.2f, %.2f]",
                    measured_distance,
                    minimum_valid_distance,
                    maximum_valid_distance);

                pid_rate.sleep();
                continue;
            }

            const double distance_error =
                measured_distance -
                target_distance;

            if (std::fabs(distance_error) <=
                distance_tolerance)
            {
                pid_cmd.linear.x = 0.0;
                ++stable_samples;
            }
            else
            {
                stable_samples = 0;

                // 距离大于 0.20 m 时向前；
                // 距离小于 0.20 m 时向后。
                pid_cmd.linear.x =
                    PID_Realize2(
                        &front_pid,
                        distance_error,
                        maximum_linear_speed,
                        -maximum_linear_speed,
                        0.5);
            }

            pid_cmd.linear.y = 0.0;

            double control_yaw = yaw;
            nh_.getParam("CarYaw", control_yaw);

            const double yaw_error =
                std::atan2(
                    std::sin(
                        target_map_yaw - control_yaw),
                    std::cos(
                        target_map_yaw - control_yaw));

            pid_cmd.angular.z =
                Limit_Value(
                    Kp_yaw * yaw_error,
                    maximum_angular_speed,
                    -maximum_angular_speed);

            cmd_vel_pub__.publish(pid_cmd);

            ROS_INFO(
                "Front PID: distance_qian_x=%.3f "
                "error=%.3f linear=%.3f angular=%.3f "
                "stable=%d/%d",
                measured_distance,
                distance_error,
                pid_cmd.linear.x,
                pid_cmd.angular.z,
                stable_samples,
                required_stable_samples);

            if (stable_samples >=
                required_stable_samples)
            {
                adjustment_succeeded = true;
                break;
            }

            pid_rate.sleep();
        }

        publishStop();

        if (adjustment_succeeded)
        {
            ROS_INFO(
                "distance_qian_x PID completed: "
                "front distance is stable at %.2f m",
                target_distance);
        }
        else
        {
            ROS_WARN(
                "distance_qian_x PID failed or timed out; "
                "vehicle stopped");
        }

        return adjustment_succeeded;
    };

    std::string target_warehouse = "UNKNOWN";
    target_locked_ = false;

    if (target_num == 1)
    {
        ROS_INFO(
            "Entering GotoC1 state: "
            "Real Car Warehouse Matching");

        nh_.getParam(
            "real_class",
            target_warehouse);
    }
    else
    {
        ROS_INFO(
            "Entering GotoC2 state: "
            "Sim Car Warehouse Matching");

        nh_.getParam(
            "sim_class",
            target_warehouse);
    }

    nh_.setParam(
        "auto_park_target",
        target_warehouse);
    nh_.setParam(
        "start_auto_park",
        0);
    nh_.setParam(
        "auto_park_status",
        "IDLE");

    int target_class = -1;

    if (target_warehouse.find("食品") !=
        std::string::npos)
    {
        target_class = 0;
    }
    else if (
        target_warehouse.find("日用品") !=
            std::string::npos ||
        target_warehouse.find("用品") !=
            std::string::npos)
    {
        target_class = 1;
    }
    else if (
        target_warehouse.find("电子") !=
            std::string::npos ||
        target_warehouse.find("电") !=
            std::string::npos ||
        target_warehouse.find("生产") !=
            std::string::npos)
    {
        target_class = 2;
    }

    if (target_class < 0)
    {
        ROS_ERROR(
            "Unknown target warehouse: %s",
            target_warehouse.c_str());
    }

    enum ObservationWall
    {
        TOP_WALL = 0,
        RIGHT_WALL,
        BOTTOM_WALL,
        LEFT_WALL
    };

    struct Pose
    {
        double x;
        double y;
        double yaw;
        int wall;
        int first_slot;
        int last_slot;
        bool has_secondary_view;
        int secondary_wall;
        int secondary_first_slot;
        int secondary_last_slot;
        double secondary_yaw;
        bool fallback;
    };

    typedef std::vector<Pose> ObservationRegion;

    // 生产区左上内墙角约为 (-2.0, -1.3)，格宽 0.5 m，
    // 共 10 x 4 格。全部保留为参数，便于现场根据实际地图整体平移。
    const double production_left_x =
        nh_.param("production_left_x", -2.0);
    const double production_top_y =
        nh_.param("production_top_y", -1.3);
    const double production_cell_size =
        std::max(
            0.10,
            nh_.param("production_cell_size", 0.5));
    const int production_columns =
        std::max(
            1,
            nh_.param("production_columns", 10));
    const int production_rows =
        std::max(
            1,
            nh_.param("production_rows", 4));
    const double primary_view_distance =
        std::max(
            0.10,
            nh_.param(
                "warehouse_primary_view_distance",
                1.00));
    const double fallback_view_distance =
        std::max(
            0.10,
            nh_.param(
                "warehouse_fallback_view_distance",
                0.50));
    const double lateral_retry_offset =
        std::min(
            production_cell_size * 0.40,
            std::max(
                0.0,
                nh_.param(
                    "warehouse_lateral_retry_offset",
                    0.15)));
    const double corner_view_inset =
        std::max(
            0.10,
            nh_.param(
                "warehouse_corner_view_inset",
                0.75));
    const double observation_navigation_timeout =
        std::max(
            1.0,
            nh_.param(
                "warehouse_observation_navigation_timeout",
                20.0));
    const double corner_turn_timeout =
        std::max(
            1.0,
            nh_.param(
                "warehouse_corner_turn_timeout",
                6.0));
    const double stable_recognition_distance =
        std::max(
            0.10,
            nh_.param(
                "warehouse_stable_recognition_distance",
                1.25));
    const double stable_half_fov =
        0.5 *
        std::min(
            179.0,
            std::max(
                1.0,
                nh_.param(
                    "warehouse_stable_fov_degrees",
                    120.0))) *
        M_PI / 180.0;
    const double settle_duration =
        std::max(
            0.0,
            nh_.param(
                "warehouse_observation_settle_duration",
                0.4));
    const double recognition_duration =
        std::max(
            0.1,
            nh_.param(
                "warehouse_observation_recognition_duration",
                2.0));
    const double recognition_message_max_age =
        std::max(
            0.1,
            nh_.param(
                "warehouse_recognition_message_max_age",
                2.0));

    const double production_right_x =
        production_left_x +
        production_columns * production_cell_size;
    const double production_bottom_y =
        production_top_y -
        production_rows * production_cell_size;

    auto observationWallName =
        [](int wall) -> const char *
    {
        switch (wall)
        {
        case TOP_WALL:
            return "top";
        case RIGHT_WALL:
            return "right";
        case BOTTOM_WALL:
            return "bottom";
        case LEFT_WALL:
            return "left";
        default:
            return "unknown";
        }
    };

    std::vector<ObservationRegion> search_regions;

    auto addObservationRegion =
        [&](int wall,
            int first_slot,
            int last_slot,
            double view_distance,
            bool fallback)
    {
        Pose point = {};
        point.wall = wall;
        point.first_slot = first_slot;
        point.last_slot = last_slot;
        point.has_secondary_view = false;
        point.secondary_wall = -1;
        point.secondary_first_slot = -1;
        point.secondary_last_slot = -1;
        point.secondary_yaw = 0.0;
        point.fallback = fallback;

        double tangent_x = 0.0;
        double tangent_y = 0.0;

        if (wall == TOP_WALL ||
            wall == BOTTOM_WALL)
        {
            if (first_slot < 0 ||
                last_slot < first_slot ||
                last_slot >= production_columns)
            {
                return;
            }

            point.x =
                production_left_x +
                (first_slot + last_slot + 1.0) *
                    production_cell_size / 2.0;
            tangent_x = 1.0;

            if (wall == TOP_WALL)
            {
                point.y =
                    production_top_y - view_distance;
                point.yaw = M_PI / 2.0;
            }
            else
            {
                point.y =
                    production_bottom_y + view_distance;
                point.yaw = -M_PI / 2.0;
            }
        }
        else
        {
            if (first_slot < 0 ||
                last_slot < first_slot ||
                last_slot >= production_rows)
            {
                return;
            }

            point.y =
                production_top_y -
                (first_slot + last_slot + 1.0) *
                    production_cell_size / 2.0;
            tangent_y = 1.0;

            if (wall == RIGHT_WALL)
            {
                point.x =
                    production_right_x - view_distance;
                point.yaw = 0.0;
            }
            else
            {
                point.x =
                    production_left_x + view_distance;
                point.yaw = M_PI;
            }
        }

        ObservationRegion region;
        const double candidate_shifts[3] = {
            0.0,
            -lateral_retry_offset,
            lateral_retry_offset};
        const int candidate_count =
            lateral_retry_offset > 0.0 ? 3 : 1;

        for (int candidate_index = 0;
             candidate_index < candidate_count;
             ++candidate_index)
        {
            Pose candidate = point;
            candidate.x +=
                candidate_shifts[candidate_index] *
                tangent_x;
            candidate.y +=
                candidate_shifts[candidate_index] *
                tangent_y;
            region.push_back(candidate);
        }

        search_regions.push_back(region);
    };

    // 角落只使用一个共享位置，但分别朝向相邻的两面墙停车识别。
    auto addCornerObservationRegion =
        [&](int primary_wall,
            int primary_first_slot,
            int primary_last_slot,
            int secondary_wall,
            int secondary_first_slot,
            int secondary_last_slot,
            double point_x,
            double point_y,
            double primary_yaw,
            double secondary_yaw,
            bool fallback)
    {
        Pose point = {};
        point.x = point_x;
        point.y = point_y;
        point.yaw = primary_yaw;
        point.wall = primary_wall;
        point.first_slot = primary_first_slot;
        point.last_slot = primary_last_slot;
        point.has_secondary_view = true;
        point.secondary_wall = secondary_wall;
        point.secondary_first_slot =
            secondary_first_slot;
        point.secondary_last_slot =
            secondary_last_slot;
        point.secondary_yaw = secondary_yaw;
        point.fallback = fallback;

        // 沿角平分线的切向给出两个候选位置， nominal 点仍优先。
        const double bisector_x =
            std::cos(primary_yaw) +
            std::cos(secondary_yaw);
        const double bisector_y =
            std::sin(primary_yaw) +
            std::sin(secondary_yaw);
        const double bisector_length =
            std::hypot(bisector_x, bisector_y);
        const double tangent_x =
            -bisector_y / bisector_length;
        const double tangent_y =
            bisector_x / bisector_length;

        ObservationRegion region;
        const double candidate_shifts[3] = {
            0.0,
            -lateral_retry_offset,
            lateral_retry_offset};
        const int candidate_count =
            lateral_retry_offset > 0.0 ? 3 : 1;

        for (int candidate_index = 0;
             candidate_index < candidate_count;
             ++candidate_index)
        {
            Pose candidate = point;
            candidate.x +=
                candidate_shifts[candidate_index] *
                tangent_x;
            candidate.y +=
                candidate_shifts[candidate_index] *
                tangent_y;
            region.push_back(candidate);
        }

        search_regions.push_back(region);
    };

    // 墙面垂直观察距离和稳定识别上限均为 1.25 m；
    // 中段每三个墙格共用一个位置，四角由共享位置覆盖相邻两面墙。
    auto addCoarseWall =
        [&](int wall,
            int first_slot,
            int last_slot,
            bool reverse)
    {
        if (first_slot > last_slot)
        {
            return;
        }

        struct SlotRange
        {
            int first;
            int last;
        };

        std::vector<SlotRange> ranges;

        for (int first = first_slot;
             first <= last_slot;
             first += 3)
        {
            SlotRange range;
            range.first = first;
            range.last =
                std::min(
                    first + 2,
                    last_slot);
            ranges.push_back(range);
        }

        if (reverse)
        {
            for (std::vector<SlotRange>::reverse_iterator it =
                     ranges.rbegin();
                 it != ranges.rend();
                 ++it)
            {
                addObservationRegion(
                    wall,
                    it->first,
                    it->last,
                    primary_view_distance,
                    false);
            }
        }
        else
        {
            for (std::vector<SlotRange>::const_iterator it =
                     ranges.begin();
                 it != ranges.end();
                 ++it)
            {
                addObservationRegion(
                    wall,
                    it->first,
                    it->last,
                    primary_view_distance,
                    false);
            }
        }
    };

    const int horizontal_corner_span =
        std::min(2, production_columns);
    const int vertical_corner_span =
        std::min(2, production_rows);

    // 快速阶段按顺时针绕场一周。默认 10 x 4 格时共 8 个位置：
    // 四个共享角点，以及上下长墙中段各两个位置。
    addCornerObservationRegion(
        LEFT_WALL,
        0,
        vertical_corner_span - 1,
        TOP_WALL,
        0,
        horizontal_corner_span - 1,
        production_left_x + corner_view_inset,
        production_top_y - corner_view_inset,
        M_PI,
        M_PI / 2.0,
        false);

    addCoarseWall(
        TOP_WALL,
        horizontal_corner_span,
        production_columns - horizontal_corner_span - 1,
        false);

    addCornerObservationRegion(
        TOP_WALL,
        production_columns - horizontal_corner_span,
        production_columns - 1,
        RIGHT_WALL,
        0,
        vertical_corner_span - 1,
        production_right_x - corner_view_inset,
        production_top_y - corner_view_inset,
        M_PI / 2.0,
        0.0,
        false);

    addCoarseWall(
        RIGHT_WALL,
        vertical_corner_span,
        production_rows - vertical_corner_span - 1,
        false);

    addCornerObservationRegion(
        RIGHT_WALL,
        production_rows - vertical_corner_span,
        production_rows - 1,
        BOTTOM_WALL,
        production_columns - horizontal_corner_span,
        production_columns - 1,
        production_right_x - corner_view_inset,
        production_bottom_y + corner_view_inset,
        0.0,
        -M_PI / 2.0,
        false);

    addCoarseWall(
        BOTTOM_WALL,
        horizontal_corner_span,
        production_columns - horizontal_corner_span - 1,
        true);

    addCornerObservationRegion(
        BOTTOM_WALL,
        0,
        horizontal_corner_span - 1,
        LEFT_WALL,
        production_rows - vertical_corner_span,
        production_rows - 1,
        production_left_x + corner_view_inset,
        production_bottom_y + corner_view_inset,
        -M_PI / 2.0,
        M_PI,
        false);

    addCoarseWall(
        LEFT_WALL,
        vertical_corner_span,
        production_rows - vertical_corner_span - 1,
        true);

    const std::size_t coarse_region_count =
        search_regions.size();

    // 未找到时逐个检查物理边界格。角落格只生成一个共享位置，
    // 默认共有 24 个位置，而不是将四个角按两面墙重复计算成 28 个。
    addCornerObservationRegion(
        LEFT_WALL,
        0,
        0,
        TOP_WALL,
        0,
        0,
        production_left_x + fallback_view_distance,
        production_top_y - fallback_view_distance,
        M_PI,
        M_PI / 2.0,
        true);

    for (int slot = 1;
         slot < production_columns - 1;
         ++slot)
    {
        addObservationRegion(
            TOP_WALL,
            slot,
            slot,
            fallback_view_distance,
            true);
    }

    addCornerObservationRegion(
        TOP_WALL,
        production_columns - 1,
        production_columns - 1,
        RIGHT_WALL,
        0,
        0,
        production_right_x - fallback_view_distance,
        production_top_y - fallback_view_distance,
        M_PI / 2.0,
        0.0,
        true);

    for (int slot = 1;
         slot < production_rows - 1;
         ++slot)
    {
        addObservationRegion(
            RIGHT_WALL,
            slot,
            slot,
            fallback_view_distance,
            true);
    }

    addCornerObservationRegion(
        RIGHT_WALL,
        production_rows - 1,
        production_rows - 1,
        BOTTOM_WALL,
        production_columns - 1,
        production_columns - 1,
        production_right_x - fallback_view_distance,
        production_bottom_y + fallback_view_distance,
        0.0,
        -M_PI / 2.0,
        true);

    for (int slot = production_columns - 2;
         slot >= 1;
         --slot)
    {
        addObservationRegion(
            BOTTOM_WALL,
            slot,
            slot,
            fallback_view_distance,
            true);
    }

    addCornerObservationRegion(
        BOTTOM_WALL,
        0,
        0,
        LEFT_WALL,
        production_rows - 1,
        production_rows - 1,
        production_left_x + fallback_view_distance,
        production_bottom_y + fallback_view_distance,
        -M_PI / 2.0,
        M_PI,
        true);

    for (int slot = production_rows - 2;
         slot >= 1;
         --slot)
    {
        addObservationRegion(
            LEFT_WALL,
            slot,
            slot,
            fallback_view_distance,
            true);
    }

    ROS_INFO(
        "Generated %zu warehouse observation regions: "
        "%zu coarse regions followed by per-slot fallback; "
        "grid=%dx%d cell=%.2f "
        "left=%.2f right=%.2f top=%.2f bottom=%.2f",
        search_regions.size(),
        coarse_region_count,
        production_columns,
        production_rows,
        production_cell_size,
        production_left_x,
        production_right_x,
        production_top_y,
        production_bottom_y);

    auto getCurrentMapPose =
        [this](double &car_x,
               double &car_y,
               double &car_yaw) -> bool
    {
        return nh_.getParam("CarX", car_x) &&
               nh_.getParam("CarY", car_y) &&
               nh_.getParam("CarYaw", car_yaw) &&
               std::isfinite(car_x) &&
               std::isfinite(car_y) &&
               std::isfinite(car_yaw);
    };

    auto getWallSlotCenter =
        [&](int wall,
            int slot,
            double &target_x,
            double &target_y)
    {
        if (wall == TOP_WALL ||
            wall == BOTTOM_WALL)
        {
            target_x =
                production_left_x +
                (slot + 0.5) * production_cell_size;
            target_y =
                wall == TOP_WALL
                    ? production_top_y
                    : production_bottom_y;
        }
        else
        {
            target_x =
                wall == RIGHT_WALL
                    ? production_right_x
                    : production_left_x;
            target_y =
                production_top_y -
                (slot + 0.5) * production_cell_size;
        }
    };

    auto lineOfSightClear =
        [](const nav_msgs::OccupancyGrid::ConstPtr &costmap,
           double start_x,
           double start_y,
           double target_x,
           double target_y) -> bool
    {
        if (!costmap ||
            costmap->info.resolution <= 0.0 ||
            costmap->info.width == 0 ||
            costmap->info.height == 0)
        {
            return true;
        }

        const double dx = target_x - start_x;
        const double dy = target_y - start_y;
        const double distance = std::hypot(dx, dy);

        if (distance <= 1e-6)
        {
            return true;
        }

        // 起点附近是机器人自身，终点附近是静态墙；两端不参与锥桶遮挡判断。
        const double first_sample = 0.12;
        const double last_sample =
            std::max(first_sample, distance - 0.12);
        const double sample_step =
            std::max(
                0.02,
                static_cast<double>(
                    costmap->info.resolution) /
                    2.0);

        for (double ray_distance = first_sample;
             ray_distance < last_sample;
             ray_distance += sample_step)
        {
            const double sample_x =
                start_x + dx * ray_distance / distance;
            const double sample_y =
                start_y + dy * ray_distance / distance;
            const int map_x =
                static_cast<int>(
                    std::floor(
                        (sample_x -
                         costmap->info.origin.position.x) /
                        costmap->info.resolution));
            const int map_y =
                static_cast<int>(
                    std::floor(
                        (sample_y -
                         costmap->info.origin.position.y) /
                        costmap->info.resolution));

            if (map_x < 0 ||
                map_y < 0 ||
                map_x >=
                    static_cast<int>(costmap->info.width) ||
                map_y >=
                    static_cast<int>(costmap->info.height))
            {
                return false;
            }

            const int occupancy =
                costmap->data[map_y * costmap->info.width +
                              map_x];

            // 只把致命障碍栅格当成视觉遮挡，避免墙和锥桶的膨胀区误判。
            if (occupancy >= 100)
            {
                return false;
            }
        }

        return true;
    };

    auto visibleWallSlotCount =
        [&](int wall,
            int first_slot,
            int last_slot,
            double observer_x,
            double observer_y,
            double observer_yaw,
            const nav_msgs::OccupancyGrid::ConstPtr &costmap) -> int
    {
        int visible_slots = 0;

        for (int slot = first_slot;
             slot <= last_slot;
             ++slot)
        {
            double target_x = 0.0;
            double target_y = 0.0;
            getWallSlotCenter(
                wall,
                slot,
                target_x,
                target_y);

            const double dx = target_x - observer_x;
            const double dy = target_y - observer_y;
            const double target_distance =
                std::hypot(dx, dy);

            if (target_distance >
                    stable_recognition_distance ||
                target_distance < 0.10)
            {
                continue;
            }

            const double target_bearing =
                std::atan2(dy, dx);
            const double bearing_error =
                std::atan2(
                    std::sin(
                        target_bearing -
                        observer_yaw),
                    std::cos(
                        target_bearing -
                        observer_yaw));

            if (std::fabs(bearing_error) >
                stable_half_fov)
            {
                continue;
            }

            if (!lineOfSightClear(
                    costmap,
                    observer_x,
                    observer_y,
                    target_x,
                    target_y))
            {
                continue;
            }

            ++visible_slots;
        }

        return visible_slots;
    };

    auto visibleSlotCount =
        [&](const ObservationRegion &region,
            double observer_x,
            double observer_y,
            double observer_yaw,
            const nav_msgs::OccupancyGrid::ConstPtr &costmap) -> int
    {
        return visibleWallSlotCount(
            region.front().wall,
            region.front().first_slot,
            region.front().last_slot,
            observer_x,
            observer_y,
            observer_yaw,
            costmap);
    };

    auto plannedVisibleSlotCount =
        [&](const ObservationRegion &region,
            double observer_x,
            double observer_y,
            const nav_msgs::OccupancyGrid::ConstPtr &costmap) -> int
    {
        const Pose &description = region.front();
        int visible_slots =
            visibleWallSlotCount(
                description.wall,
                description.first_slot,
                description.last_slot,
                observer_x,
                observer_y,
                description.yaw,
                costmap);

        if (description.has_secondary_view)
        {
            visible_slots +=
                visibleWallSlotCount(
                    description.secondary_wall,
                    description.secondary_first_slot,
                    description.secondary_last_slot,
                    observer_x,
                    observer_y,
                    description.secondary_yaw,
                    costmap);
        }

        return visible_slots;
    };

    ros::ServiceClient make_plan_client =
        nh_.serviceClient<nav_msgs::GetPlan>(
            "/move_base/make_plan");
    bool make_plan_available =
        make_plan_client.waitForExistence(
            ros::Duration(1.0));

    if (!make_plan_available)
    {
        ROS_WARN(
            "/move_base/make_plan is unavailable; "
            "observation candidates cannot be pre-screened");
    }

    auto candidateHasPlan =
        [&](const Pose &candidate,
            double &path_length) -> bool
    {
        path_length =
            std::numeric_limits<double>::infinity();

        if (!make_plan_available)
        {
            return true;
        }

        double car_x = 0.0;
        double car_y = 0.0;
        double car_yaw = 0.0;

        if (!getCurrentMapPose(
                car_x,
                car_y,
                car_yaw))
        {
            ROS_WARN(
                "CarX/CarY/CarYaw unavailable; "
                "using the nominal observation candidate");
            make_plan_available = false;
            return true;
        }

        nav_msgs::GetPlan plan_srv;
        plan_srv.request.start.header.frame_id = "map";
        plan_srv.request.start.header.stamp =
            ros::Time::now();
        plan_srv.request.start.pose.position.x = car_x;
        plan_srv.request.start.pose.position.y = car_y;

        tf2::Quaternion start_quaternion;
        start_quaternion.setRPY(0.0, 0.0, car_yaw);
        plan_srv.request.start.pose.orientation =
            tf2::toMsg(start_quaternion);

        plan_srv.request.goal.header.frame_id = "map";
        plan_srv.request.goal.header.stamp =
            plan_srv.request.start.header.stamp;
        plan_srv.request.goal.pose.position.x = candidate.x;
        plan_srv.request.goal.pose.position.y = candidate.y;

        tf2::Quaternion goal_quaternion;
        goal_quaternion.setRPY(0.0, 0.0, candidate.yaw);
        plan_srv.request.goal.pose.orientation =
            tf2::toMsg(goal_quaternion);
        plan_srv.request.tolerance = 0.10;

        if (!make_plan_client.call(plan_srv))
        {
            ROS_WARN(
                "/move_base/make_plan call failed; "
                "falling back to normal navigation");
            make_plan_available = false;
            return true;
        }

        if (plan_srv.response.plan.poses.empty())
        {
            return std::hypot(
                       candidate.x - car_x,
                       candidate.y - car_y) <= 0.10;
        }

        path_length = 0.0;

        for (std::size_t pose_index = 1;
             pose_index <
             plan_srv.response.plan.poses.size();
             ++pose_index)
        {
            const geometry_msgs::Point &previous =
                plan_srv.response.plan
                    .poses[pose_index - 1]
                    .pose.position;
            const geometry_msgs::Point &current =
                plan_srv.response.plan
                    .poses[pose_index]
                    .pose.position;
            path_length +=
                std::hypot(
                    current.x - previous.x,
                    current.y - previous.y);
        }

        return true;
    };

    auto selectObservationCandidate =
        [&](const ObservationRegion &region,
            Pose &selected_candidate) -> bool
    {
        const nav_msgs::OccupancyGrid::ConstPtr costmap =
            ros::topic::waitForMessage<
                nav_msgs::OccupancyGrid>(
                "/move_base/global_costmap/costmap",
                nh_,
                ros::Duration(0.4));

        int best_clear_slots = -1;
        double best_path_length =
            std::numeric_limits<double>::infinity();
        bool candidate_found = false;
        const int region_slot_count =
            region.front().last_slot -
            region.front().first_slot +
            1 +
            (region.front().has_secondary_view
                 ? region.front().secondary_last_slot -
                       region.front().secondary_first_slot +
                       1
                 : 0);

        for (std::size_t candidate_index = 0;
             candidate_index < region.size();
             ++candidate_index)
        {
            const Pose &candidate =
                region[candidate_index];
            const int geometrically_visible_slots =
                plannedVisibleSlotCount(
                    region,
                    candidate.x,
                    candidate.y,
                    nav_msgs::OccupancyGrid::ConstPtr());

            if (geometrically_visible_slots == 0)
            {
                continue;
            }

            const int clear_slots =
                plannedVisibleSlotCount(
                    region,
                    candidate.x,
                    candidate.y,
                    costmap);

            double path_length = 0.0;

            if (!candidateHasPlan(
                    candidate,
                    path_length))
            {
                continue;
            }

            if (!candidate_found ||
                clear_slots > best_clear_slots ||
                (clear_slots == best_clear_slots &&
                 path_length < best_path_length))
            {
                selected_candidate = candidate;
                best_clear_slots = clear_slots;
                best_path_length = path_length;
                candidate_found = true;
            }

            // 标准位置可达且覆盖整个区域时，无需继续请求另外两个规划。
            if (candidate_index == 0 &&
                clear_slots == region_slot_count)
            {
                break;
            }
        }

        if (candidate_found)
        {
            if (selected_candidate.has_secondary_view)
            {
                ROS_INFO(
                    "Selected corner observation candidate: "
                    "%s[%d-%d] + %s[%d-%d] "
                    "clear=%d/%d x=%.3f y=%.3f",
                    observationWallName(
                        selected_candidate.wall),
                    selected_candidate.first_slot + 1,
                    selected_candidate.last_slot + 1,
                    observationWallName(
                        selected_candidate.secondary_wall),
                    selected_candidate.secondary_first_slot + 1,
                    selected_candidate.secondary_last_slot + 1,
                    best_clear_slots,
                    region_slot_count,
                    selected_candidate.x,
                    selected_candidate.y);
            }
            else
            {
                ROS_INFO(
                    "Selected observation candidate: "
                    "wall=%s slots=%d-%d "
                    "clear=%d/%d x=%.3f y=%.3f yaw=%.3f",
                    observationWallName(
                        selected_candidate.wall),
                    selected_candidate.first_slot + 1,
                    selected_candidate.last_slot + 1,
                    best_clear_slots,
                    region_slot_count,
                    selected_candidate.x,
                    selected_candidate.y,
                    selected_candidate.yaw);
            }
        }

        return candidate_found;
    };

    auto navigateToSearchPoint =
        [this,
         &publishStop](
            const Pose &point,
            int point_number,
            const char *point_kind,
            double timeout) -> bool
    {
        sendPos(
            point.x,
            point.y,
            point.yaw);

        const bool finished_before_timeout =
            ac_.waitForResult(
                ros::Duration(timeout));
        const actionlib::SimpleClientGoalState navigation_state =
            ac_.getState();

        if (finished_before_timeout &&
            navigation_state ==
                actionlib::SimpleClientGoalState::SUCCEEDED)
        {
            return true;
        }

        if (!finished_before_timeout)
        {
            ac_.cancelGoal();
            ac_.waitForResult(
                ros::Duration(0.5));
        }

        publishStop();

        ROS_WARN(
            "%s observation point %d navigation failed: "
            "finished=%s state=%s",
            point_kind,
            point_number,
            finished_before_timeout ? "true" : "false",
            navigation_state.toString().c_str());

        return false;
    };

    auto currentPoseCanObserve =
        [&](const ObservationRegion &region,
            Pose &actual_observation_pose) -> bool
    {
        double car_x = 0.0;
        double car_y = 0.0;
        double car_yaw = 0.0;

        if (!getCurrentMapPose(
                car_x,
                car_y,
                car_yaw))
        {
            return false;
        }

        const nav_msgs::OccupancyGrid::ConstPtr costmap =
            ros::topic::waitForMessage<
                nav_msgs::OccupancyGrid>(
                "/move_base/global_costmap/costmap",
                nh_,
                ros::Duration(0.4));
        const int geometrically_visible_slots =
            visibleSlotCount(
                region,
                car_x,
                car_y,
                car_yaw,
                nav_msgs::OccupancyGrid::ConstPtr());

        if (geometrically_visible_slots == 0)
        {
            return false;
        }

        const int clear_slots =
            visibleSlotCount(
                region,
                car_x,
                car_y,
                car_yaw,
                costmap);

        actual_observation_pose = region.front();
        actual_observation_pose.x = car_x;
        actual_observation_pose.y = car_y;
        actual_observation_pose.yaw = car_yaw;

        ROS_WARN(
            "Navigation did not succeed, but the stopped "
            "vehicle is within the stable view of %d slot(s) "
            "(%d currently clear) in "
            "wall=%s slots=%d-%d",
            geometrically_visible_slots,
            clear_slots,
            observationWallName(region.front().wall),
            region.front().first_slot + 1,
            region.front().last_slot + 1);

        return true;
    };

    const int navigation_max_attempts = 3;
    const double navigation_retry_delay = 1.0;

    auto navigateWithRetry =
        [this,
         &publishStop,
         navigation_max_attempts,
         navigation_retry_delay](
            double goal_x,
            double goal_y,
            double goal_yaw,
            double timeout,
            const char *goal_name) -> bool
    {
        for (int attempt = 1;
             attempt <= navigation_max_attempts && ros::ok();
             ++attempt)
        {
            ROS_INFO(
                "Navigating to %s, attempt %d/%d",
                goal_name,
                attempt,
                navigation_max_attempts);

            sendPos(goal_x, goal_y, goal_yaw);

            const bool finished_before_timeout =
                ac_.waitForResult(ros::Duration(timeout));
            const actionlib::SimpleClientGoalState navigation_state =
                ac_.getState();

            if (finished_before_timeout &&
                navigation_state ==
                    actionlib::SimpleClientGoalState::SUCCEEDED)
            {
                ROS_INFO(
                    "Reached %s on attempt %d/%d",
                    goal_name,
                    attempt,
                    navigation_max_attempts);
                return true;
            }

            if (!finished_before_timeout)
            {
                ac_.cancelGoal();
                ac_.waitForResult(ros::Duration(1.0));
            }

            publishStop();

            ROS_WARN(
                "Failed to reach %s on attempt %d/%d: "
                "finished=%s state=%s",
                goal_name,
                attempt,
                navigation_max_attempts,
                finished_before_timeout ? "true" : "false",
                navigation_state.toString().c_str());

            if (attempt < navigation_max_attempts)
            {
                ros::WallDuration(
                    navigation_retry_delay)
                    .sleep();
            }
        }

        return false;
    };

    // 已获得目标物的有效坐标。
    bool target_found = false;

    // 只有该变量为 true，实车阶段才进行播报。
    bool ready_to_announce = false;

    double target_dx = 0.0;
    double target_dy = 0.0;
    double target_line_a = 0.0;

    if (target_class >= 0 &&
        warehouse_parking_cache_[target_class].valid &&
        ros::ok())
    {
        WarehouseParkingCache &cached =
            warehouse_parking_cache_[target_class];

        ROS_INFO(
            "Trying cached warehouse parking pose: "
            "class=%d x=%.3f y=%.3f yaw=%.3f",
            target_class,
            cached.parking_x,
            cached.parking_y,
            cached.parking_yaw);

        const bool reached_cached_pose =
            navigateWithRetry(
                cached.parking_x,
                cached.parking_y,
                cached.parking_yaw,
                15.0,
                "cached warehouse parking pose");

        if (reached_cached_pose)
        {
            // 横向对正只会在收到该类别的新 RKNN 结果后成功，
            // 因而同时作为缓存位置的现场复核。
            const bool cache_verified =
                adjustLateralPosition(
                    false,
                    cached.parking_yaw,
                    target_class);

            if (cache_verified)
            {
                const bool front_adjusted =
                    adjustFrontDistance(
                        false,
                        cached.parking_yaw);

                last_parking_observation_x_ =
                    cached.observation_x;
                last_parking_observation_y_ =
                    cached.observation_y;
                last_parking_observation_yaw_ =
                    cached.observation_yaw;
                last_parking_observation_valid_ = true;

                target_found = true;
                ready_to_announce = true;

                nh_.setParam(
                    "auto_park_status",
                    front_adjusted
                        ? "DONE"
                        : "FAILED");

                if (front_adjusted)
                {
                    ROS_INFO(
                        "Cached warehouse parking pose "
                        "verified and parking completed");
                }
                else
                {
                    ROS_WARN(
                        "Cached warehouse pose was verified, "
                        "but front-distance adjustment failed");
                }
            }
            else
            {
                cached.valid = false;
                nh_.setParam(
                    "auto_park_status",
                    "IDLE");

                ROS_WARN(
                    "Cached warehouse pose could not be "
                    "verified by RKNN; invalidate it and "
                    "fall back to observation search");
            }
        }
        else
        {
            cached.valid = false;

            ROS_WARN(
                "Cached warehouse pose is unreachable; "
                "invalidate it and fall back to "
                "observation search");
        }

        target_locked_ = false;
        publishStop();
    }

    for (int i = 0;
         i < static_cast<int>(search_regions.size()) &&
         !target_found &&
         ros::ok();
         ++i)
    {
        const ObservationRegion &observation_region =
            search_regions[i];
        const Pose &region_description =
            observation_region.front();
        const char *observation_phase =
            region_description.fallback
                ? "Fallback"
                : "Coarse";

        if (region_description.has_secondary_view)
        {
            ROS_INFO(
                "Selecting %s corner observation "
                "region %d/%zu: %s[%d-%d] + %s[%d-%d]",
                observation_phase,
                i + 1,
                search_regions.size(),
                observationWallName(
                    region_description.wall),
                region_description.first_slot + 1,
                region_description.last_slot + 1,
                observationWallName(
                    region_description.secondary_wall),
                region_description.secondary_first_slot + 1,
                region_description.secondary_last_slot + 1);
        }
        else
        {
            ROS_INFO(
                "Selecting %s warehouse observation "
                "region %d/%zu: wall=%s slots=%d-%d",
                observation_phase,
                i + 1,
                search_regions.size(),
                observationWallName(
                    region_description.wall),
                region_description.first_slot + 1,
                region_description.last_slot + 1);
        }

        Pose active_observation_point = {};

        if (!selectObservationCandidate(
                observation_region,
                active_observation_point))
        {
            ROS_WARN(
                "%s observation region %d has no "
                "currently reachable, visible candidate; "
                "skip without sending a navigation goal",
                observation_phase,
                i + 1);
            continue;
        }

        // 行驶期间的识别结果一律作废。
        current_signal_class_ = -1;
        last_signal_class_time_ = ros::Time(0);
        last_signal_detection_time_ = ros::Time(0);
        target_locked_ = false;

        bool reached_observation =
            navigateToSearchPoint(
                active_observation_point,
                i + 1,
                observation_phase,
                observation_navigation_timeout);

        // 无论 action 成功还是失败，都先停车并等待画面稳定；
        // 绝不使用车辆运动过程中的识别结果。
        publishStop();
        ros::Duration(settle_duration).sleep();

        const bool can_observe =
            reached_observation ||
            currentPoseCanObserve(
                observation_region,
                active_observation_point);

        if (!can_observe)
        {
            ROS_WARN(
                "Observation region %d was not reached and "
                "the stopped vehicle is outside its stable "
                "view; continue with the next region",
                i + 1);
            continue;
        }

        ROS_DEBUG(
            "Vehicle stopped in valid region %d, "
            "starting stop-and-look signal search",
            i + 1);

        // 普通位置只观察一面墙；共享角点原地转向后再观察相邻墙。
        const int view_count =
            region_description.has_secondary_view
                ? 2
                : 1;

        bool leave_observation_point = false;
        ros::Rate rate(20);

        for (int view = 0;
             view < view_count && ros::ok();
             ++view)
        {
            if (view == 1)
            {
                // 转向期间的所有识别结果均作废。
                current_signal_class_ = -1;
                last_signal_class_time_ = ros::Time(0);
                last_signal_detection_time_ = ros::Time(0);
                target_locked_ = false;

                Pose secondary_observation =
                    active_observation_point;
                secondary_observation.yaw =
                    region_description.secondary_yaw;

                const bool secondary_turn_succeeded =
                    navigateToSearchPoint(
                        secondary_observation,
                        i + 1,
                        "Corner second-wall turn",
                        corner_turn_timeout);

                publishStop();
                ros::Duration(settle_duration).sleep();

                bool secondary_view_ready =
                    secondary_turn_succeeded;

                if (!secondary_view_ready)
                {
                    double car_x = 0.0;
                    double car_y = 0.0;
                    double car_yaw = 0.0;

                    if (getCurrentMapPose(
                            car_x,
                            car_y,
                            car_yaw) &&
                        visibleWallSlotCount(
                            region_description.secondary_wall,
                            region_description.secondary_first_slot,
                            region_description.secondary_last_slot,
                            car_x,
                            car_y,
                            car_yaw,
                            nav_msgs::OccupancyGrid::ConstPtr()) > 0)
                    {
                        secondary_observation.x = car_x;
                        secondary_observation.y = car_y;
                        secondary_observation.yaw = car_yaw;
                        secondary_view_ready = true;
                    }
                }

                if (!secondary_view_ready)
                {
                    ROS_WARN(
                        "Corner region %d could not obtain a "
                        "stable stopped view of the second wall",
                        i + 1);
                    break;
                }

                active_observation_point =
                    secondary_observation;
            }

            // 只使用停车以后产生的新识别结果。
            current_signal_class_ = -1;
            last_signal_class_time_ = ros::Time(0);
            last_signal_detection_time_ = ros::Time(0);
            target_locked_ = false;

            ROS_DEBUG(
                "Observation region %d, static view %d/%d "
                "wall=%s yaw=%.3f",
                i + 1,
                view + 1,
                view_count,
                observationWallName(
                    view == 0
                        ? region_description.wall
                        : region_description.secondary_wall),
                active_observation_point.yaw);

            ros::Time recognition_start =
                ros::Time::now();

            while (ros::ok() &&
                   (ros::Time::now() -
                    recognition_start)
                           .toSec() <
                       recognition_duration)
            {
                ros::spinOnce();

                bool class_recent =
                    !last_signal_class_time_.isZero() &&
                    (ros::Time::now() -
                     last_signal_class_time_)
                            .toSec() <
                        recognition_message_max_age;

                bool detection_recent =
                    !last_signal_detection_time_.isZero() &&
                    (ros::Time::now() -
                     last_signal_detection_time_)
                            .toSec() <
                        recognition_message_max_age;

                if (class_recent &&
                    detection_recent &&
                    current_signal_class_ ==
                        target_class)
                {
                    ROS_INFO(
                        "Target class matched. "
                        "class=%d center_x=%.1f",
                        current_signal_class_,
                        signal_center_x_);

                    publishStop();

                    // 车辆在进入本轮识别前已经停车并清空旧结果，
                    // 因此首次匹配即可锁定；不再做会导致消息过期的
                    // 重复确认。
                    locked_signal_class_ =
                        current_signal_class_;
                    locked_center_x_ =
                        signal_center_x_;
                    locked_box_x_l_ =
                        signal_box_x_l_;
                    locked_box_x_r_ =
                        signal_box_x_r_;

                    target_locked_ = true;

                    ROS_INFO(
                        "Target locked at stopped observation. "
                        "class=%d center=%.1f "
                        "left=%.1f right=%.1f",
                        locked_signal_class_,
                        locked_center_x_,
                        locked_box_x_l_,
                        locked_box_x_r_);

                    ourgoal::getLaserPoint srv;

                    srv.request.center_x =
                        std::max(
                            0,
                            std::min(
                                639,
                                static_cast<int>(
                                    std::round(
                                        locked_center_x_))));

                    srv.request.left_x =
                        std::max(
                            0,
                            std::min(
                                639,
                                static_cast<int>(
                                    std::round(
                                        locked_box_x_l_))));

                    srv.request.right_x =
                        std::max(
                            0,
                            std::min(
                                639,
                                static_cast<int>(
                                    std::round(
                                        locked_box_x_r_))));

                    srv.request.mode = false;

                    if (srv.request.left_x <
                            srv.request.right_x &&
                        vision_gettool_client_.call(srv))
                    {
                        target_dx =
                            (srv.response.dx_left +
                             srv.response.dx_right) /
                            2.0;

                        target_dy =
                            (srv.response.dy_left +
                             srv.response.dy_right) /
                            2.0;

                        target_line_a =
                            srv.response.line_a;

                        ROS_INFO(
                            "Signal metric center: "
                            "dx=%.3f dy=%.3f line_a=%.3f",
                            target_dx,
                            target_dy,
                            target_line_a);

                        nh_.setParam(
                            "signal_target_dx",
                            target_dx);
                        nh_.setParam(
                            "signal_target_dy",
                            target_dy);
                        nh_.setParam(
                            "signal_target_line_a",
                            target_line_a);

                        point_2d target_in_base;
                        target_in_base.x = target_dx;
                        target_in_base.y = target_dy;

                        double car_x =
                            nh_.param("CarX", 0.0);
                        double car_y =
                            nh_.param("CarY", 0.0);
                        double car_yaw =
                            nh_.param("CarYaw", 0.0);

                        point_2d target_in_map =
                            rotate(
                                target_in_base,
                                car_yaw);

                        target_in_map =
                            translate(
                                target_in_map,
                                car_x,
                                car_y);

                        // 标牌一定在四面墙之一。用目标点到四面墙的
                        // 距离确定所属墙面，并将停泊朝向锁定为正方向，
                        // 避免角落斜视或激光直线拟合误差造成斜停。
                        const int viewed_wall =
                            view == 0
                                ? region_description.wall
                                : region_description.secondary_wall;
                        const double wall_distances[4] = {
                            std::fabs(
                                target_in_map.y -
                                production_top_y),
                            std::fabs(
                                target_in_map.x -
                                production_right_x),
                            std::fabs(
                                target_in_map.y -
                                production_bottom_y),
                            std::fabs(
                                target_in_map.x -
                                production_left_x)};

                        int parking_wall = viewed_wall;

                        for (int wall = TOP_WALL;
                             wall <= LEFT_WALL;
                             ++wall)
                        {
                            if (wall_distances[wall] <
                                wall_distances[parking_wall])
                            {
                                parking_wall = wall;
                            }
                        }

                        double target_yaw =
                            active_observation_point.yaw;

                        switch (parking_wall)
                        {
                        case TOP_WALL:
                            target_yaw = M_PI / 2.0;
                            break;
                        case RIGHT_WALL:
                            target_yaw = 0.0;
                            break;
                        case BOTTOM_WALL:
                            target_yaw = -M_PI / 2.0;
                            break;
                        case LEFT_WALL:
                            target_yaw = M_PI;
                            break;
                        }

                        // 先计算距离目标物 0.30 m 的导航停车点。
                        const double stop_distance = 0.30;

                        point_2d map_target;

                        map_target.x =
                            target_in_map.x -
                            stop_distance *
                                std::cos(target_yaw);

                        map_target.y =
                            target_in_map.y -
                            stop_distance *
                                std::sin(target_yaw);

                        const double original_goal_x =
                            map_target.x;
                        const double original_goal_y =
                            map_target.y;

                        // 将导航目标限制在生产区向内缩 0.30 m 的
                        // 矩形内；右下格仍可到达约 x=2.70。
                        const double min_goal_x =
                            production_left_x + stop_distance;
                        const double max_goal_x =
                            production_right_x - stop_distance;
                        const double min_goal_y =
                            production_bottom_y + stop_distance;
                        const double max_goal_y =
                            production_top_y - stop_distance;

                        if (map_target.x < min_goal_x)
                        {
                            map_target.x = min_goal_x;
                        }
                        else if (map_target.x > max_goal_x)
                        {
                            map_target.x = max_goal_x;
                        }

                        if (map_target.y < min_goal_y)
                        {
                            map_target.y = min_goal_y;
                        }
                        else if (map_target.y > max_goal_y)
                        {
                            map_target.y = max_goal_y;
                        }

                        const bool goal_clamped =
                            map_target.x != original_goal_x ||
                            map_target.y != original_goal_y;

                        // 坐标限幅只移动停车点，绝不改变墙面固定朝向。
                        const double face_target_dx =
                            target_in_map.x -
                            map_target.x;

                        const double face_target_dy =
                            target_in_map.y -
                            map_target.y;

                        const double actual_stop_distance =
                            std::hypot(
                                face_target_dx,
                                face_target_dy);

                        ROS_INFO(
                            "Warehouse target in map: "
                            "x=%.3f y=%.3f "
                            "viewed_wall=%s parking_wall=%s",
                            target_in_map.x,
                            target_in_map.y,
                            observationWallName(viewed_wall),
                            observationWallName(parking_wall));

                        if (goal_clamped)
                        {
                            ROS_WARN(
                                "Parking goal clamped: "
                                "(%.3f, %.3f) -> "
                                "(%.3f, %.3f)",
                                original_goal_x,
                                original_goal_y,
                                map_target.x,
                                map_target.y);
                        }

                        ROS_INFO(
                            "Parking goal in map: "
                            "x=%.3f y=%.3f yaw=%.3f "
                            "distance_to_target=%.3f",
                            map_target.x,
                            map_target.y,
                            target_yaw,
                            actual_stop_distance);

                        // 保存本次发现目标时所在的观察点。
                        // 仅供后续 GotoD 导航巡线粗起点失败时恢复。
                        last_parking_observation_x_ =
                            active_observation_point.x;
                        last_parking_observation_y_ =
                            active_observation_point.y;
                        last_parking_observation_yaw_ =
                            active_observation_point.yaw;
                        last_parking_observation_valid_ = true;

                        ROS_INFO(
                            "Saved GotoC observation point for "
                            "GotoD coarse-goal recovery: "
                            "x=%.3f y=%.3f yaw=%.3f",
                            last_parking_observation_x_,
                            last_parking_observation_y_,
                            last_parking_observation_yaw_);

                        // 目标已经找到，此后不再搜索其他观测点。
                        target_found = true;

                        bool navigation_succeeded = false;

                        while (ros::ok() &&
                               !navigation_succeeded)
                        {
                            navigation_succeeded =
                                navigateWithRetry(
                                    map_target.x,
                                    map_target.y,
                                    target_yaw,
                                    15.0,
                                    "warehouse parking pose");

                            if (navigation_succeeded)
                            {
                                break;
                            }

                            ROS_ERROR(
                                "Warehouse parking navigation failed "
                                "after %d attempts; returning to "
                                "observation point %d",
                                navigation_max_attempts,
                                i + 1);

                            bool returned_to_observation = false;

                            while (ros::ok() &&
                                   !returned_to_observation)
                            {
                                returned_to_observation =
                                    navigateWithRetry(
                                        active_observation_point.x,
                                        active_observation_point.y,
                                        active_observation_point.yaw,
                                        20.0,
                                        "GotoC observation point");

                                if (!returned_to_observation)
                                {
                                    ROS_ERROR(
                                        "Failed to return to GotoC "
                                        "observation point; retrying");
                                    ros::WallDuration(
                                        navigation_retry_delay)
                                        .sleep();
                                }
                            }

                            if (returned_to_observation)
                            {
                                ROS_WARN(
                                    "Returned to observation point %d; "
                                    "retry warehouse parking navigation",
                                    i + 1);
                            }
                        }

                        if (!navigation_succeeded)
                        {
                            publishStop();
                            target_locked_ = false;
                            return;
                        }

                        ROS_INFO(
                            "Arrived at warehouse navigation pose");

                        if (target_class >= 0)
                        {
                            WarehouseParkingCache &cached =
                                warehouse_parking_cache_[target_class];

                            cached.valid = true;
                            cached.parking_x = map_target.x;
                            cached.parking_y = map_target.y;
                            cached.parking_yaw = target_yaw;
                            cached.observation_x =
                                active_observation_point.x;
                            cached.observation_y =
                                active_observation_point.y;
                            cached.observation_yaw =
                                active_observation_point.yaw;

                            ROS_INFO(
                                "Cached warehouse pose: "
                                "class=%d parking=(%.3f, %.3f, %.3f) "
                                "observation=(%.3f, %.3f, %.3f)",
                                target_class,
                                cached.parking_x,
                                cached.parking_y,
                                cached.parking_yaw,
                                cached.observation_x,
                                cached.observation_y,
                                cached.observation_yaw);
                        }

                        bool lateral_adjusted = false;
                        bool front_adjusted = false;

                        // 先根据停车后的新 RKNN 结果横向对正，
                        // 再调整与目标之间的前后距离。
                        lateral_adjusted =
                            adjustLateralPosition(
                                false,
                                target_yaw,
                                target_class);

                        front_adjusted =
                            adjustFrontDistance(
                                false,
                                target_yaw);

                        if (lateral_adjusted &&
                            front_adjusted)
                        {
                            nh_.setParam(
                                "auto_park_status",
                                "DONE");

                            ready_to_announce = true;

                            ROS_INFO(
                                "Warehouse parking completed "
                                "using RKNN lateral alignment "
                                "and distance_qian_x PID");
                        }
                        else
                        {
                            nh_.setParam(
                                "auto_park_status",
                                "FAILED");

                            // 导航已经成功；最终位置微调即使失败，
                            // 也在微调流程结束后允许播报。
                            ready_to_announce = true;

                            ROS_WARN(
                                "Navigation succeeded, but "
                                "final position adjustment "
                                "failed (lateral=%s front=%s); "
                                "continue with announcement",
                                lateral_adjusted
                                    ? "true"
                                    : "false",
                                front_adjusted
                                    ? "true"
                                    : "false");
                        }

                        target_locked_ = false;
                        leave_observation_point = true;
                        break;
                    }
                    else
                    {
                        ROS_WARN(
                            "Target class matched, but "
                            "/srv_getLaserPoint failed");

                        target_locked_ = false;
                    }
                }

                rate.sleep();
            }

            if (target_found ||
                leave_observation_point)
            {
                break;
            }
        }

        publishStop();

        if (target_found)
        {
            break;
        }

        ROS_WARN(
            "Target not found in observation region %d, go next",
            i + 1);
    }

    if (target_found &&
        ready_to_announce &&
        target_num == 1)
    {
        std::string item;
        std::string room;

        nh_.getParam("real_item", item);
        nh_.getParam("real_room", room);

        char tts_cmd[512];

        std::snprintf(
            tts_cmd,
            sizeof(tts_cmd),
            "espeak -v zh+f2 "
            "\"已将%s放入%s\" -s 130",
            item.c_str(),
            room.c_str());

        system(tts_cmd);
    }

    if (!target_found)
    {
        ROS_ERROR(
            "Failed to find target warehouse "
            "after all observation regions");

        nh_.setParam(
            "auto_park_status",
            "FAILED");
    }

    current_state =
        (target_num == 1)
            ? GOTOC2_
            : Gazebo_;
}

// =========================================================================
// 仿真协同 (包含播报3)
// =========================================================================
void OURSWITCH::Gazebo()
{
    ROS_INFO("Entering Gazebo state: remote PC simulation");

    nh_.setParam("gazebo_sim_done", 0);
    nh_.setParam("start_gazebo_sim", 1);

    int sim_done = 0;
    ros::Rate rate(10);

    ROS_INFO("Waiting for remote PC gazebo_success...");

    while (sim_done == 0 && ros::ok())
    {
        nh_.getParam("gazebo_sim_done", sim_done);
        ros::spinOnce();
        rate.sleep();
    }

    nh_.setParam("start_gazebo_sim", 0);

    ROS_INFO("Remote PC simulation finished.");

    std::string sim_item = "UNKNOWN";
    std::string sim_room = "UNKNOWN";
    nh_.getParam("sim_item", sim_item);
    nh_.getParam("sim_room", sim_room);

    char tts_cmd[512];
    sprintf(tts_cmd, "espeak -v zh+f2 \"仿真任务已完成，已将%s放入%s\" -s 130",
            sim_item.c_str(), sim_room.c_str());
    system(tts_cmd);

    current_state = GOTOD_;
}

// =========================================================================
// 交通决策与路径选择
// =========================================================================
void OURSWITCH::GotoD()
{
    ROS_INFO("Entering GotoD state: Traffic Light Detection");

    nh_.setParam("start_traffic_light_det", 0);

    const int max_attempts =
        std::max(1, nh_.param("gotod_nav_max_attempts", 3));
    const double retry_delay =
        std::max(0.0, nh_.param("gotod_nav_retry_delay", 1.0));

    auto navigateWithRetry =
        [this, max_attempts, retry_delay](
            double goal_x,
            double goal_y,
            double goal_yaw,
            double timeout,
            const char *goal_name) -> bool
    {
        for (int attempt = 1;
             attempt <= max_attempts && ros::ok();
             ++attempt)
        {
            ROS_INFO(
                "Navigating to %s, attempt %d/%d",
                goal_name,
                attempt,
                max_attempts);

            sendPos(goal_x, goal_y, goal_yaw);

            const bool finished_before_timeout =
                ac_.waitForResult(ros::Duration(timeout));
            const actionlib::SimpleClientGoalState navigation_state =
                ac_.getState();

            if (finished_before_timeout &&
                navigation_state ==
                    actionlib::SimpleClientGoalState::SUCCEEDED)
            {
                ROS_INFO(
                    "Reached %s on attempt %d/%d",
                    goal_name,
                    attempt,
                    max_attempts);
                return true;
            }

            if (!finished_before_timeout)
            {
                ac_.cancelGoal();
                ac_.waitForResult(ros::Duration(1.0));
            }

            geometry_msgs::Twist stop_cmd;
            cmd_vel_pub__.publish(stop_cmd);

            ROS_WARN(
                "Failed to reach %s on attempt %d/%d: "
                "finished=%s state=%s",
                goal_name,
                attempt,
                max_attempts,
                finished_before_timeout ? "true" : "false",
                navigation_state.toString().c_str());

            if (attempt < max_attempts)
            {
                ros::WallDuration(retry_delay).sleep();
            }
        }

        return false;
    };

    auto recoverToLastParkingObservation =
        [this, &navigateWithRetry]() -> bool
    {
        if (!last_parking_observation_valid_)
        {
            goto_d_recovery_pending_ = false;

            ROS_ERROR(
                "Cannot recover GotoD coarse-goal navigation: "
                "no GotoC observation point was saved");
            return false;
        }

        goto_d_recovery_pending_ = true;

        ROS_WARN(
            "Returning to the last GotoC observation point "
            "before retrying the GotoD coarse goal: "
            "x=%.3f y=%.3f yaw=%.3f",
            last_parking_observation_x_,
            last_parking_observation_y_,
            last_parking_observation_yaw_);

        const bool recovered =
            navigateWithRetry(
                last_parking_observation_x_,
                last_parking_observation_y_,
                last_parking_observation_yaw_,
                20.0,
                "last GotoC observation point");

        if (recovered)
        {
            goto_d_recovery_pending_ = false;
            ROS_INFO(
                "Returned to the last GotoC observation point; "
                "the GotoD coarse goal can be retried");
        }
        else
        {
            ROS_ERROR(
                "Failed to return to the last GotoC observation "
                "point; keep coarse-goal recovery pending");
        }

        return recovered;
    };

    // 如果上一轮连观察点也未能返回，本轮先完成恢复，
    // 暂不尝试巡线粗起点。
    if (goto_d_recovery_pending_ &&
        !recoverToLastParkingObservation())
    {
        geometry_msgs::Twist stop_cmd;
        cmd_vel_pub__.publish(stop_cmd);
        current_state = GOTOD_;
        ros::WallDuration(retry_delay).sleep();
        return;
    }

    const bool reached_fixed_point =
        navigateWithRetry(
            0.4,
            -3.05,
            -1.57,
            20.0,
            "fixed GotoD point");

    if (!reached_fixed_point)
    {
        geometry_msgs::Twist stop_cmd;
        cmd_vel_pub__.publish(stop_cmd);

        ROS_ERROR(
            "Fixed GotoD point failed after %d attempts. "
            "Return to the last GotoC observation point "
            "before retrying the coarse goal.",
            max_attempts);

        const bool recovered =
            recoverToLastParkingObservation();

        if (recovered)
        {
            ROS_WARN(
                "Coarse-goal recovery completed; retry the fixed "
                "GotoD point on the next state-machine cycle");
        }

        current_state = GOTOD_;
        ros::WallDuration(retry_delay).sleep();
        return;
    }

    // 粗起点已经成功到达，从这里开始不再启用观察点恢复机制。
    goto_d_recovery_pending_ = false;

    ROS_INFO(
        "Arrived at the coarse line-following start; "
        "skip gap alignment.");

    nh_.setParam("start_traffic_light_det", 1);

    current_state = VISION_LINE_;
}

// =========================================================================
// 视觉巡线到达终点 (包含播报4)
// =========================================================================
void OURSWITCH::vision_line()
{
    ROS_INFO("Entering VISION_LINE state");

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

