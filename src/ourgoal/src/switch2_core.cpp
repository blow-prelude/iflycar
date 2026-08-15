#include "switch2.h"

#include <chrono>
#include <limits>
#include <thread>

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
