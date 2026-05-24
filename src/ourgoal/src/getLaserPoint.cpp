
#include <ros/ros.h>
#include "camera_2d_lidar_calibration/vision_points.h"
#include "ourgoal/getLaserPoint.h"
#include <iostream>
#include <vector>

double LaserPoints_array[3][650];

ros::Subscriber sub_laserPoint;
ros::ServiceServer server_getLaser;
void getLaserXYZ_from_vision(const camera_2d_lidar_calibration::vision_points::ConstPtr& LaserPoints_msg)
{
    for (size_t i = 0; i < LaserPoints_msg->dx.size(); ++i)
    {
       LaserPoints_array[0][i] = LaserPoints_msg->dx[i];
       LaserPoints_array[1][i] = LaserPoints_msg->dy[i];
       LaserPoints_array[2][i] = LaserPoints_msg->dz[i];
       if(LaserPoints_msg->dx[i]>0){
            printf("i:%d\n",i);
            printf("dx:%lf,dy:%lf\n\n",LaserPoints_msg->dx[i],LaserPoints_msg->dy[i]);
       }
    }
}

struct LaserPoint
{
    double x,y,offset;
};
std::vector<LaserPoint> points ;


LaserPoint getLeftPoint_fromIn(int x_id)
{
    LaserPoint res;

    int off_id = 1;
    double tmp_dx;
    while(233)
    {
        tmp_dx = LaserPoints_array[0][x_id-off_id];
        if(tmp_dx>0)        
        {
            // 偏左
            res.x = LaserPoints_array[0][x_id-off_id];
            res.y = LaserPoints_array[1][x_id-off_id];
            res.offset = -1 * off_id;
            break;
        }
        ++off_id;
    }
    return res;
}
LaserPoint getRightPoint_fromIn(int x_id)
{
    LaserPoint res;

    int off_id = 1;
    double tmp_dx;
    while(233)
    {
        tmp_dx = LaserPoints_array[0][x_id+off_id];

        if(tmp_dx>0)        
        {
            // 偏左
            res.x = LaserPoints_array[0][x_id+off_id];
            res.y = LaserPoints_array[1][x_id+off_id];
            res.offset = 1 * off_id;
            break;
        }
        ++off_id;
    }
    return res;
}

LaserPoint getLeftPoint_fromOut(int x_id,int offset)
{
    LaserPoint res;

    int off_id = 1 * offset;
    double tmp_dx;
    while(233)
    {
        tmp_dx = LaserPoints_array[0][x_id+off_id];

        if(tmp_dx>0)        
        {
            // 偏左
            res.x = LaserPoints_array[0][x_id+off_id];
            res.y = LaserPoints_array[1][x_id+off_id];
            res.offset = 1 * off_id;
            break;
        }
        ++off_id;
    }
    return res;
}
LaserPoint getRightPoint_fromOut(int x_id,int offset)
{
    LaserPoint res;

    int off_id = 1 * offset;
    double tmp_dx;
    while(233)
    {
        tmp_dx = LaserPoints_array[0][x_id-off_id];

        if(tmp_dx>0)        
        {
            // 偏左
            res.x = LaserPoints_array[0][x_id-off_id];
            res.y = LaserPoints_array[1][x_id-off_id];
            res.offset = -1 * off_id;
            break;
        }
        ++off_id;
    }
    return res;
}
void addVector(int l,int r)
{
    points.clear();
    printf("l:%d,r:%d\n",l,r);
    for(int i=l;i<=r;i++)
    {
        if(LaserPoints_array[0][i]>0)
        {
            LaserPoint t;
            t.x = LaserPoints_array[0][i];
            t.y = LaserPoints_array[1][i];
            t.offset = i;
            points.push_back(t);
        }
    }
}
void fitLine(const std::vector<LaserPoint>& points, double& a, double& b) {
    double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0;
    int n = points.size();

    for (const auto& point : points) {
        sumX += point.x;
        sumY += point.y;
        sumXY += point.x * point.y;
        sumX2 += point.x * point.x;
        //printf("(%.2f,%.2f)\n",point.x,point.y);
    }

    double xMean = sumX / n;
    double yMean = sumY / n;
    if( (n * sumX2 - sumX * sumX) != 0)
    {
        a = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        b = yMean - a * xMean;
    }
    else
    {
        a = 0xff;
        b = 0xff;
    }
    printf("a:%.2f , b:%.2f, n:%d\n",a,b,n);
}
bool doReq(ourgoal::getLaserPoint::Request& req,
          ourgoal::getLaserPoint::Response& resp)
{
    int left_x = req.left_x,right_x = req.right_x,center_x = req.center_x;
    int size = right_x-left_x;
    double mode = req.mode;
    
    LaserPoint L,R;

    if(mode) 
    {
        L = getLeftPoint_fromIn(center_x);
        R = getRightPoint_fromIn(center_x);
    }
    else
    {
        L = getLeftPoint_fromOut(left_x,size/6);
        R = getRightPoint_fromOut(right_x,size/6);
    }
    
    double a, b;
    addVector(L.offset + left_x, R.offset + right_x);
    fitLine(points, a, b);
    
    resp.dx_left = L.x;
    resp.dy_left = L.y;
    resp.offset_left = L.offset;
    ROS_INFO("LX:%f", L.x);
    ROS_INFO("LY:%f", L.y);
    ROS_INFO("LO:%d", L.offset);
    resp.dx_right = R.x;
    resp.dy_right = R.y;
    resp.offset_right = R.offset;
    ROS_INFO("RX:%f", R.x);
    ROS_INFO("RY:%f", R.y);
    ROS_INFO("RO:%d", R.offset);
    resp.line_a = a;
    resp.line_b = b; //y=ax+b
    return true;
}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "getLaserPoint_node");
    ros::NodeHandle nh("~");
    printf("getLaserPoint_node run\n");

    sub_laserPoint = nh.subscribe("/vision_points", 50,getLaserXYZ_from_vision);
    server_getLaser = nh.advertiseService("/srv_getLaserPoint",doReq); //  /srv_getLaserPoint
    //

    ros::spin();
    return 0;
}
