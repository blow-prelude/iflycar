#include "switch2.h"

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

    speakText(
        std::string("仿真任务已完成，已将") +
        sim_item +
        "放入" +
        sim_room);

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
            0.3,
            -3.02,
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

    geometry_msgs::Twist stop_cmd;
    cmd_vel_pub__.publish(stop_cmd);
    ros::WallDuration(0.3).sleep();

    const bool gap_alignment_enabled =
        nh_.param("gotod_gap_alignment_enabled", true);

    bool gap_aligned = false;
    if (gap_alignment_enabled)
    {
        // 墙线相对最终 base_link 原点的有符号前向距离：
        // 正值表示仍在墙内，0 表示对齐墙线，负值表示越过墙线。
        const double target_gap_distance =
            nh_.param("gotod_gap_target_distance", -0.10);
        const double alignment_timeout =
            std::max(1.0, nh_.param("gotod_gap_alignment_timeout", 8.0));
        const double sample_spread_limit =
            std::max(0.005, nh_.param("gotod_gap_control_sample_spread", 0.04));
        const double position_tolerance =
            std::max(0.005, nh_.param("gotod_gap_position_tolerance", 0.01));
        const double maximum_linear_speed =
            std::max(0.01, nh_.param("gotod_gap_max_linear_speed", 0.08));
        const double maximum_correction =
            std::max(
                position_tolerance,
                nh_.param("gotod_gap_max_correction", 0.60));
        const double heading_tolerance = 3.0 * M_PI / 180.0;
        const double maximum_angular_speed = 0.15;
        const int required_stable_updates = 3;

        auto median = [](std::vector<double> values)
        {
            std::sort(values.begin(), values.end());
            const size_t middle = values.size() / 2;
            if (values.size() % 2 == 0)
                return (values[middle - 1] + values[middle]) / 2.0;
            return values[middle];
        };

        auto spread = [](const std::vector<double> &values)
        {
            const std::pair<std::vector<double>::const_iterator,
                            std::vector<double>::const_iterator>
                bounds = std::minmax_element(values.begin(), values.end());
            return *bounds.second - *bounds.first;
        };

        {
            std::lock_guard<std::mutex> lock(gap_mutex_);
            gap_mid_x_samples_.clear();
            gap_mid_y_samples_.clear();
            gap_width_samples_.clear();
            collect_gap_samples_ = true;
        }

        ROS_INFO(
            "Collecting stationary lidar scans from inside the wall...");

        bool target_locked = false;
        size_t processed_sample_count = 0;
        double target_odom_x = 0.0;
        double target_odom_y = 0.0;
        double hold_yaw = 0.0;
        double detected_gap_width = 0.0;
        ros::WallRate scan_rate(20.0);
        const ros::WallTime detection_start = ros::WallTime::now();

        while (ros::ok() &&
               (ros::WallTime::now() - detection_start).toSec() <
                   gap_detection_timeout_)
        {
            cmd_vel_pub__.publish(stop_cmd);

            std::vector<double> mid_x_samples;
            std::vector<double> mid_y_samples;
            std::vector<double> width_samples;

            {
                std::lock_guard<std::mutex> lock(gap_mutex_);
                const size_t sample_count = gap_mid_x_samples_.size();
                if (sample_count >=
                        static_cast<size_t>(gap_required_samples_) &&
                    sample_count != processed_sample_count)
                {
                    const size_t first_sample =
                        sample_count - gap_required_samples_;
                    mid_x_samples.assign(
                        gap_mid_x_samples_.begin() + first_sample,
                        gap_mid_x_samples_.end());
                    mid_y_samples.assign(
                        gap_mid_y_samples_.begin() + first_sample,
                        gap_mid_y_samples_.end());
                    width_samples.assign(
                        gap_width_samples_.begin() + first_sample,
                        gap_width_samples_.end());
                    processed_sample_count = sample_count;
                }
            }

            if (mid_x_samples.empty())
            {
                scan_rate.sleep();
                continue;
            }

            const bool samples_stable =
                spread(mid_x_samples) <= sample_spread_limit &&
                spread(mid_y_samples) <= sample_spread_limit &&
                spread(width_samples) <= sample_spread_limit;

            if (!samples_stable)
            {
                ROS_WARN_THROTTLE(
                    1.0,
                    "GotoD stationary gap samples are unstable");
                scan_rate.sleep();
                continue;
            }

            const double gap_laser_x = median(mid_x_samples);
            const double gap_laser_y = median(mid_y_samples);
            detected_gap_width = median(width_samples);

            const double lidar_cos = std::cos(lidar_yaw_);
            const double lidar_sin = std::sin(lidar_yaw_);
            const double gap_base_x =
                lidar_offset_x_ +
                lidar_cos * gap_laser_x -
                lidar_sin * gap_laser_y;
            const double gap_base_y =
                lidar_offset_y_ +
                lidar_sin * gap_laser_x +
                lidar_cos * gap_laser_y;
            const double forward_correction =
                gap_base_x - target_gap_distance;
            const double lateral_correction = gap_base_y;
            const double correction_distance =
                std::hypot(forward_correction, lateral_correction);

            // 必须放在安全修正判断之前，确保修正被拒绝时也能看到
            // 雷达扫描到的墙线距离和实际需要移动的距离。
            ROS_INFO(
                "GotoD wall-line scan: lidar_forward=%.3f m "
                "base_link_forward=%.3f m lateral_offset=%.3f m "
                "target_signed_distance=%.3f m "
                "required_forward_travel=%.3f m "
                "required_planar_travel=%.3f m",
                gap_laser_x,
                gap_base_x,
                gap_base_y,
                target_gap_distance,
                forward_correction,
                correction_distance);

            if (correction_distance > maximum_correction)
            {
                ROS_ERROR(
                    "Reject unsafe GotoD gap correction: "
                    "distance=%.3f m exceeds %.3f m",
                    correction_distance,
                    maximum_correction);
                break;
            }

            const bool odom_fresh =
                odom_received_ &&
                !last_odom_wall_time_.isZero() &&
                (ros::WallTime::now() - last_odom_wall_time_).toSec() < 0.3;
            if (!odom_fresh)
            {
                ROS_WARN_THROTTLE(
                    1.0,
                    "Fresh odometry is unavailable; cannot lock the "
                    "stationary lidar result");
                scan_rate.sleep();
                continue;
            }

            hold_yaw = yaw;
            const double odom_cos = std::cos(hold_yaw);
            const double odom_sin = std::sin(hold_yaw);
            target_odom_x =
                odom_x_ + odom_cos * forward_correction -
                odom_sin * lateral_correction;
            target_odom_y =
                odom_y_ + odom_sin * forward_correction +
                odom_cos * lateral_correction;
            target_locked = true;

            double car_x = 0.0;
            double car_y = 0.0;
            double car_yaw = 0.0;
            if (nh_.getParam("CarX", car_x) &&
                nh_.getParam("CarY", car_y) &&
                nh_.getParam("CarYaw", car_yaw))
            {
                const double car_cos = std::cos(car_yaw);
                const double car_sin = std::sin(car_yaw);
                const double exit_map_x =
                    car_x + car_cos * gap_base_x - car_sin * gap_base_y;
                const double exit_map_y =
                    car_y + car_sin * gap_base_x + car_cos * gap_base_y;
                nh_.setParam("gotod_detected_exit_x", exit_map_x);
                nh_.setParam("gotod_detected_exit_y", exit_map_y);
            }

            ROS_INFO(
                "Stationary GotoD scan locked: width=%.3f "
                "gap_base=(%.3f, %.3f) target_wall_distance=%.3f "
                "forward_correction=%.3f lateral_correction=%.3f",
                detected_gap_width,
                gap_base_x,
                gap_base_y,
                target_gap_distance,
                forward_correction,
                lateral_correction);
            break;
        }

        {
            std::lock_guard<std::mutex> lock(gap_mutex_);
            collect_gap_samples_ = false;
        }
        cmd_vel_pub__.publish(stop_cmd);

        if (target_locked)
        {
            ROS_INFO(
                "Stationary lidar sampling finished; starting chassis "
                "fine adjustment.");

            geometry_msgs::Twist alignment_cmd;
            int stable_updates = 0;
            ros::WallRate alignment_rate(20.0);
            const ros::WallTime alignment_start = ros::WallTime::now();

            while (ros::ok() &&
                   (ros::WallTime::now() - alignment_start).toSec() <
                       alignment_timeout)
            {
                const bool odom_fresh =
                    odom_received_ &&
                    !last_odom_wall_time_.isZero() &&
                    (ros::WallTime::now() - last_odom_wall_time_).toSec() <
                        0.3;

                if (!odom_fresh)
                {
                    alignment_cmd = geometry_msgs::Twist();
                    stable_updates = 0;
                    ROS_WARN_THROTTLE(
                        1.0,
                        "Fresh odometry is unavailable during GotoD "
                        "fine adjustment");
                }
                else
                {
                    const double odom_error_x = target_odom_x - odom_x_;
                    const double odom_error_y = target_odom_y - odom_y_;
                    const double odom_cos = std::cos(yaw);
                    const double odom_sin = std::sin(yaw);
                    const double forward_error =
                        odom_cos * odom_error_x + odom_sin * odom_error_y;
                    const double lateral_error =
                        -odom_sin * odom_error_x + odom_cos * odom_error_y;
                    const double heading_error =
                        std::atan2(
                            std::sin(hold_yaw - yaw),
                            std::cos(hold_yaw - yaw));

                    alignment_cmd.linear.x =
                        Limit_Value(
                            Kp_dist * forward_error,
                            maximum_linear_speed,
                            -maximum_linear_speed);
                    alignment_cmd.linear.y =
                        Limit_Value(
                            Kp_dist * lateral_error,
                            maximum_linear_speed,
                            -maximum_linear_speed);
                    alignment_cmd.angular.z =
                        Limit_Value(
                            Kp_yaw * heading_error,
                            maximum_angular_speed,
                            -maximum_angular_speed);

                    const bool within_tolerance =
                        std::fabs(forward_error) <= position_tolerance &&
                        std::fabs(lateral_error) <= position_tolerance &&
                        std::fabs(heading_error) <= heading_tolerance;

                    if (within_tolerance)
                    {
                        alignment_cmd = geometry_msgs::Twist();
                        ++stable_updates;
                    }
                    else
                    {
                        stable_updates = 0;
                    }

                    ROS_INFO_THROTTLE(
                        0.5,
                        "GotoD chassis adjustment: forward_error=%.3f "
                        "lateral_error=%.3f heading_error=%.2f deg "
                        "stable=%d/%d",
                        forward_error,
                        lateral_error,
                        heading_error * 180.0 / M_PI,
                        stable_updates,
                        required_stable_updates);

                    if (stable_updates >= required_stable_updates)
                    {
                        gap_aligned = true;
                        break;
                    }
                }

                cmd_vel_pub__.publish(alignment_cmd);
                alignment_rate.sleep();
            }

            cmd_vel_pub__.publish(stop_cmd);
        }

        if (gap_aligned)
        {
            ROS_INFO(
                "Stationary-scan GotoD adjustment completed with "
                "centimeter-level tolerances.");
        }
        else
        {
            ROS_WARN(
                "Stationary-scan GotoD adjustment failed or timed out; "
                "continue from the current stopped position.");
        }
    }
    else
    {
        ROS_WARN("GotoD lidar gap alignment is disabled by parameter.");
    }

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

    speakText("任务完成");

    ROS_INFO("ALL TASKS COMPLETED SUCCESSFULLY! SHUTTING DOWN.");
    ros::shutdown();
}
