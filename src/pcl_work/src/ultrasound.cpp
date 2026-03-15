#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_normal_sphere.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/transforms.h>

#include <pcl_work/ultrasound.h>

ros::Publisher pub_ultra;

void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    float minX_dx,minX_dy,minY_dx,minY_dy,minX=0xff,minY=0xff;

    float hou_dx,hou_dy,hou_minY=0xff;
    float left_dx,left_dy,left_minX=0xff;

    for (size_t i = 0; i < cloud->points.size (); ++i)
    {
        float dx = cloud->points[i].x ;
        float dy = cloud->points[i].y ;
        if(std::fabs(dx)<std::fabs(minX))
        {
            if(dy<0)
            {
                minX=dx;
                minX_dx=dx;
                minX_dy=dy;
            }
        }
        if(std::fabs(dx)<std::fabs(left_minX))
        {
            if(dy>0)
            {
                left_minX=dx;
                left_dx=dx;
                left_dy=dy;
            }
        }
        if(std::fabs(dy)<std::fabs(minY))
        {
            if(dx>0)
            {
                minY=dy;
                minY_dx=dx;
                minY_dy=dy;
            }
        }

        if(std::fabs(dy)<std::fabs(hou_minY))
        {
            if(dx<0)
            {
                hou_minY=dy;
                hou_dx=dx;
                hou_dy=dy;
            }
        }
    }
    pcl_work::ultrasound pub_msg;
    pub_msg.distance_qian_x=minY_dx;
    pub_msg.distance_qian_y=minY_dy;
    pub_msg.distance_you_x=minX_dx;
    pub_msg.distance_you_y=minX_dy;
    pub_msg.distance_hou_x=hou_dx;
    pub_msg.distance_hou_y=hou_dy;
    pub_msg.distance_zuo_x=left_dx;
    pub_msg.distance_zuo_y=left_dy;
    pub_ultra.publish(pub_msg);

}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "ultrasound");
    ros::NodeHandle nh("~");
    ros::Subscriber sub_cluster = nh.subscribe<sensor_msgs::PointCloud2>("/cloud", 1, pointCloudCallback);
    pub_ultra = nh.advertise<pcl_work::ultrasound>("ultra",64);

    ros::spin();
    return 0;

}
