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
            -3.00,
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
