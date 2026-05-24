#include "ros/ros.h"
#include "laser_geometry/laser_geometry.h"
#include "sensor_msgs/LaserScan.h"
#include "sensor_msgs/PointCloud.h"
#include "tf/transform_listener.h"

laser_geometry::LaserProjection  projector;
ros::Publisher pub;
boost::shared_ptr<tf::TransformListener> listener_;

void chatterCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
{
    sensor_msgs::PointCloud2 cloud;
    projector.projectLaser(*msg, cloud);
    pub.publish(cloud);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laser2plc");
    ros::NodeHandle nh;


    listener_ = boost::make_shared<tf::TransformListener>();
    // listener_->setTransformTolerance(ros::Duration(0.1))
    
    pub = nh.advertise<sensor_msgs::PointCloud2>("cloud",10);
    ros::Subscriber sub = nh.subscribe("/scan", 1000, chatterCallback);   // 先有发布的scan话题

    ros::spin();
    return 0;
}