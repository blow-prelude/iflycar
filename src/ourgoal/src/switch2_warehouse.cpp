#include "switch2.h"
#include <limits>
#include <nav_msgs/GetPlan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <ros/topic.h>

// =========================================================================
// 抵达实体与仿真观测点 (包含播报2)
// =========================================================================
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
        target_warehouse.find("日用") !=
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
    const int observation_cache_confirmations =
        std::max(
            1,
            nh_.param(
                "warehouse_observation_cache_confirmations",
                2));
    const double observation_cache_center_tolerance =
        std::max(
            0.0,
            nh_.param(
                "warehouse_observation_cache_center_tolerance_px",
                120.0));
    const double observation_cache_timestamp_skew =
        std::max(
            0.0,
            nh_.param(
                "warehouse_observation_cache_timestamp_skew",
                0.50));

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

    // 上墙中段每两个墙格共用一个位置，其余墙面中段仍每三个墙格
    // 共用一个位置；四角由共享位置覆盖相邻两面墙。
    auto addCoarseWall =
        [&](int wall,
            int first_slot,
            int last_slot,
            int slots_per_position,
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
        slots_per_position =
            std::max(1, slots_per_position);

        for (int first = first_slot;
             first <= last_slot;
             first += slots_per_position)
        {
            SlotRange range;
            range.first = first;
            range.last =
                std::min(
                    first + slots_per_position - 1,
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

    // 快速阶段按顺时针绕场一周。默认 10 x 4 格时共 9 个位置：
    // 四个共享角点、上墙中段三个位置和下墙中段两个位置。
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
        2,
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
        3,
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
        3,
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
        3,
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

    // 类别观察缓存只作为一次“快速复核”入口，和停车缓存严格分离。
    // 缓存姿态失效时不阻塞原有的完整观察点搜索。
    bool cached_observation_region_inserted = false;

    if (target_class >= 0 &&
        warehouse_observation_cache_[target_class].valid)
    {
        WarehouseObservationCache &cached_observation =
            warehouse_observation_cache_[target_class];

        const int maximum_slot =
            (cached_observation.wall == TOP_WALL ||
             cached_observation.wall == BOTTOM_WALL)
                ? production_columns - 1
                : production_rows - 1;

        const bool cache_geometry_valid =
            cached_observation.wall >= TOP_WALL &&
            cached_observation.wall <= LEFT_WALL &&
            cached_observation.first_slot >= 0 &&
            cached_observation.last_slot >=
                cached_observation.first_slot &&
            cached_observation.last_slot <= maximum_slot &&
            std::isfinite(
                cached_observation.observation_x) &&
            std::isfinite(
                cached_observation.observation_y) &&
            std::isfinite(
                cached_observation.observation_yaw);

        if (!cache_geometry_valid)
        {
            cached_observation.valid = false;
            ROS_WARN(
                "Discarding invalid warehouse observation cache "
                "for class=%d",
                target_class);
        }
        else
        {
            Pose cached_pose = {};
            cached_pose.x =
                cached_observation.observation_x;
            cached_pose.y =
                cached_observation.observation_y;
            cached_pose.yaw =
                cached_observation.observation_yaw;
            cached_pose.wall = cached_observation.wall;
            cached_pose.first_slot =
                cached_observation.first_slot;
            cached_pose.last_slot =
                cached_observation.last_slot;
            cached_pose.has_secondary_view = false;
            cached_pose.secondary_wall = -1;
            cached_pose.secondary_first_slot = -1;
            cached_pose.secondary_last_slot = -1;
            cached_pose.secondary_yaw = 0.0;
            cached_pose.fallback = false;

            ObservationRegion cached_region;
            cached_region.push_back(cached_pose);
            search_regions.insert(
                search_regions.begin(),
                cached_region);
            cached_observation_region_inserted = true;

            ROS_INFO(
                "Prepended cached warehouse observation for "
                "class=%d: wall=%s slots=%d-%d "
                "pose=(%.3f, %.3f, %.3f)",
                target_class,
                observationWallName(cached_pose.wall),
                cached_pose.first_slot + 1,
                cached_pose.last_slot + 1,
                cached_pose.x,
                cached_pose.y,
                cached_pose.yaw);
        }
    }

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

    auto cacheObservationForClass =
        [&](int detected_class,
            const Pose &observation_pose,
            int wall,
            int first_slot,
            int last_slot)
    {
        if (detected_class < 0 ||
            detected_class >= 3 ||
            wall < TOP_WALL ||
            wall > LEFT_WALL ||
            first_slot < 0 ||
            last_slot < first_slot ||
            !std::isfinite(observation_pose.x) ||
            !std::isfinite(observation_pose.y) ||
            !std::isfinite(observation_pose.yaw))
        {
            return;
        }

        WarehouseObservationCache &cached_observation =
            warehouse_observation_cache_[detected_class];

        // 第一次稳定看到该类别的位置优先保留；如果它后来失效，
        // 复核失败路径会清除 valid，之后可由完整搜索写入新位置。
        if (cached_observation.valid)
        {
            return;
        }

        cached_observation.valid = true;
        cached_observation.observation_x =
            observation_pose.x;
        cached_observation.observation_y =
            observation_pose.y;
        cached_observation.observation_yaw =
            observation_pose.yaw;
        cached_observation.wall = wall;
        cached_observation.first_slot = first_slot;
        cached_observation.last_slot = last_slot;

        ROS_INFO(
            "Cached warehouse observation for class=%d: "
            "wall=%s slots=%d-%d pose=(%.3f, %.3f, %.3f)",
            detected_class,
            observationWallName(wall),
            first_slot + 1,
            last_slot + 1,
            observation_pose.x,
            observation_pose.y,
            observation_pose.yaw);
    };

    auto makeCurrentObservationPose =
        [&](const Pose &active_pose,
            int wall,
            int first_slot,
            int last_slot) -> Pose
    {
        Pose cache_pose = active_pose;
        cache_pose.wall = wall;
        cache_pose.first_slot = first_slot;
        cache_pose.last_slot = last_slot;
        cache_pose.has_secondary_view = false;
        cache_pose.secondary_wall = -1;
        cache_pose.secondary_first_slot = -1;
        cache_pose.secondary_last_slot = -1;
        cache_pose.secondary_yaw = 0.0;
        cache_pose.fallback = false;

        double car_x = 0.0;
        double car_y = 0.0;
        double car_yaw = 0.0;

        // 导航目标只是期望姿态；缓存优先使用停车后读取到的实际地图姿态。
        if (getCurrentMapPose(car_x, car_y, car_yaw))
        {
            cache_pose.x = car_x;
            cache_pose.y = car_y;
            cache_pose.yaw = car_yaw;
        }

        return cache_pose;
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
        const bool using_cached_observation =
            cached_observation_region_inserted && i == 0;
        const ObservationRegion &observation_region =
            search_regions[i];
        const Pose &region_description =
            observation_region.front();
        const char *observation_phase =
            using_cached_observation
                ? "Cached"
                : (region_description.fallback
                       ? "Fallback"
                       : "Coarse");

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
            if (using_cached_observation)
            {
                warehouse_observation_cache_[target_class].valid =
                    false;
                ROS_WARN(
                    "Cached observation pose is not currently "
                    "visible/reachable; invalidate it and continue "
                    "with the normal search");
            }

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
            if (using_cached_observation)
            {
                warehouse_observation_cache_[target_class].valid =
                    false;
                ROS_WARN(
                    "Cached observation pose was not reached and "
                    "cannot provide a stable view; invalidate it");
            }

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

            // 在静止画面中对同一类别做连续确认，避免把一帧 OCR
            // 抖动误写成类别位置缓存。
            int consecutive_observed_class = -1;
            int consecutive_observation_frames = 0;
            double consecutive_observation_center_x = -1.0;
            ros::Time last_counted_detection_time(0);
            bool observation_cache_written[3] = {
                false,
                false,
                false};

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

                const bool new_detection =
                    detection_recent &&
                    last_signal_detection_time_ !=
                        last_counted_detection_time;

                if (new_detection)
                {
                    last_counted_detection_time =
                        last_signal_detection_time_;

                    const int observed_class =
                        current_signal_class_;
                    const bool timestamps_are_close =
                        !last_signal_class_time_.isZero() &&
                        !last_signal_detection_time_.isZero() &&
                        std::fabs(
                            (last_signal_class_time_ -
                             last_signal_detection_time_)
                                .toSec()) <=
                            observation_cache_timestamp_skew;
                    const bool valid_observation =
                        class_recent &&
                        timestamps_are_close &&
                        observed_class >= 0 &&
                        observed_class < 3 &&
                        std::isfinite(signal_center_x_) &&
                        signal_center_x_ >= 0.0 &&
                        signal_center_x_ <= 640.0;

                    if (!valid_observation)
                    {
                        consecutive_observed_class = -1;
                        consecutive_observation_frames = 0;
                        consecutive_observation_center_x = -1.0;
                    }
                    else
                    {
                        const bool same_stable_class =
                            observed_class ==
                                consecutive_observed_class &&
                            std::fabs(
                                signal_center_x_ -
                                consecutive_observation_center_x) <=
                                observation_cache_center_tolerance;

                        if (same_stable_class)
                        {
                            ++consecutive_observation_frames;
                        }
                        else
                        {
                            consecutive_observed_class =
                                observed_class;
                            consecutive_observation_frames = 1;
                        }

                        consecutive_observation_center_x =
                            signal_center_x_;

                        if (consecutive_observation_frames >=
                                observation_cache_confirmations &&
                            !observation_cache_written[observed_class])
                        {
                            const int observed_wall =
                                view == 0
                                    ? region_description.wall
                                    : region_description.secondary_wall;
                            const int observed_first_slot =
                                view == 0
                                    ? region_description.first_slot
                                    : region_description
                                          .secondary_first_slot;
                            const int observed_last_slot =
                                view == 0
                                    ? region_description.last_slot
                                    : region_description
                                          .secondary_last_slot;
                            const Pose observed_pose =
                                makeCurrentObservationPose(
                                    active_observation_point,
                                    observed_wall,
                                    observed_first_slot,
                                    observed_last_slot);

                            cacheObservationForClass(
                                observed_class,
                                observed_pose,
                                observed_wall,
                                observed_first_slot,
                                observed_last_slot);
                            observation_cache_written[observed_class] =
                                true;
                        }
                    }
                }

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

                    // 目标类别已经进入既有的停车流程；同时把当前
                    // 观察姿态留作独立的快速复核缓存（若尚无该类缓存）。
                    const int target_viewed_wall =
                        view == 0
                            ? region_description.wall
                            : region_description.secondary_wall;
                    const int target_viewed_first_slot =
                        view == 0
                            ? region_description.first_slot
                            : region_description.secondary_first_slot;
                    const int target_viewed_last_slot =
                        view == 0
                            ? region_description.last_slot
                            : region_description.secondary_last_slot;
                    const Pose target_observation_pose =
                        makeCurrentObservationPose(
                            active_observation_point,
                            target_viewed_wall,
                            target_viewed_first_slot,
                            target_viewed_last_slot);
                    cacheObservationForClass(
                        current_signal_class_,
                        target_observation_pose,
                        target_viewed_wall,
                        target_viewed_first_slot,
                        target_viewed_last_slot);

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

        if (using_cached_observation && !target_found)
        {
            warehouse_observation_cache_[target_class].valid = false;
            ROS_WARN(
                "Cached observation did not reproduce target class "
                "%d; invalidate it and resume full observation search",
                target_class);
        }

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

