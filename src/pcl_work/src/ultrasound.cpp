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
    float minX_dx, minX_dy, minY_dx, minY_dy, minX = 0xff, minY = 0xff;

    float hou_dx, hou_dy, hou_minY = 0xff;
    float left_dx, left_dy, left_minX = 0xff;

    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        float dx = cloud->points[i].x;
        float dy = cloud->points[i].y;
        
        // 右侧距离计算
        if (std::fabs(dx) < std::fabs(minX))
        {
            if (dy < 0)
            {
                minX = dx;
                minX_dx = dx;
                minX_dy = dy;
            }
        }
        
        // 左侧距离计算
        if (std::fabs(dx) < std::fabs(left_minX))
        {
            if (dy > 0)
            {
                left_minX = dx;
                left_dx = dx;
                left_dy = dy;
            }
        }
        
        // 前方距离计算
        if (std::fabs(dy) < std::fabs(minY))
        {
            if (dx > 0)
            {
                minY = dy;
                minY_dx = dx;
                minY_dy = dy;
            }
        }
        
        // 后方距离计算
        if (std::fabs(dy) < std::fabs(hou_minY))
        {
            if (dx < 0)
            {
                hou_minY = dy;
                hou_dx = dx;
                hou_dy = dy;
            }
        }
    }
    
    // 填充发布消息
    pcl_work::ultrasound pub_msg;
    pub_msg.distance_qian_x = minY_dx;  // 前方x距离
    pub_msg.distance_qian_y = minY_dy;  // 前方y距离
    pub_msg.distance_you_x = minX_dx;   // 右侧x距离
    pub_msg.distance_you_y = minX_dy;   // 右侧y距离
    pub_msg.distance_hou_x = hou_dx;    // 后方x距离
    pub_msg.distance_hou_y = hou_dy;    // 后方y距离
    pub_msg.distance_zuo_x = left_dx;   // 左侧x距离
    pub_msg.distance_zuo_y = left_dy;   // 左侧y距离
    
    // 打印发布的信息（保留两位小数）
    // ROS_INFO("\n发布超声波数据:\n"
    //          "qian: x=%.2f, y=%.2f\n"
    //          "you: x=%.2f, y=%.2f\n"
    //          "hou: x=%.2f, y=%.2f\n"
    //          "zuo: x=%.2f, y=%.2f",
    //          pub_msg.distance_qian_x, pub_msg.distance_qian_y,
    //          pub_msg.distance_you_x, pub_msg.distance_you_y,
    //          pub_msg.distance_hou_x, pub_msg.distance_hou_y,
    //          pub_msg.distance_zuo_x, pub_msg.distance_zuo_y);
    
    // 发布消息
    pub_ultra.publish(pub_msg);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "ultrasound");
    // ros::NodeHandle nh("~");
    ros::NodeHandle nh;

    // 订阅点云话题
    ros::Subscriber sub_cluster = nh.subscribe<sensor_msgs::PointCloud2>("/cloud", 1, pointCloudCallback);
    
    // 发布超声波消息
    pub_ultra = nh.advertise<pcl_work::ultrasound>("ultra", 64);
    
    ROS_INFO("超声波数据发布节点已启动");
    ros::spin();
    return 0;
}
