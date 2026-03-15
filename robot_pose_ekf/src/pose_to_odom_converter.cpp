#include <ros/ros.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>

ros::Publisher odom_pub;
nav_msgs::Odometry odom_raw_tmp;
void odom_rawCallback(const nav_msgs::Odometry::ConstPtr& odom_raw_msg) {
    odom_raw_tmp.twist = odom_raw_msg->twist;
}
void poseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& pose_msg) {
    nav_msgs::Odometry odom_msg;

    odom_msg.header = pose_msg->header;
    odom_msg.pose = pose_msg->pose;
    odom_msg.child_frame_id = "base_link";
    odom_msg.twist = odom_raw_tmp.twist;

    odom_pub.publish(odom_msg);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "pose_to_odom_converter");
    ros::NodeHandle nh;

    ros::Subscriber pose_sub = nh.subscribe("robot_pose_ekf/odom_combined", 10, poseCallback);
    ros::Subscriber odom_raw_sub = nh.subscribe("/odom_raw", 10, odom_rawCallback);
    odom_pub = nh.advertise<nav_msgs::Odometry>("/odom", 10);

    ros::spin();
    return 0;
}
