#include "switch2.h"

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
        {
            double current_x = 0.0;
            const bool current_x_valid =
                nh_.getParam("CarX", current_x) && std::isfinite(current_x);

            cmd.linear.y = 0;

            if (!current_x_valid)
            {
                cmd.linear.x = 0;
                ROS_ERROR_THROTTLE(
                    1.0,
                    "GotoA case 7: CarX is unavailable; stop for safety");
                break;
            }

            // CarX >= 1.0 时仍处于坡道区域，不采用后方雷达结果，
            // 避免将斜坡误识别成后方墙面而提前停车。
            if (current_x >= 1.0)
            {
                cmd.linear.x = -max_vel;
                ROS_INFO_THROTTLE(
                    1.0,
                    "GotoA case 7: crossing ramp, CarX=%.3f",
                    current_x);
                break;
            }

            error = distance_hou_x - safe_B;
            cmd.linear.x = -Limit_Value(Kp_dist * error, max_vel, -max_vel);

            if (std::abs(error) < 0.05)
                escape_state = 8;
            break;
        }

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
