#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf/transform_datatypes.h>

// 监听 /map 坐标系到 /base_link 坐标系的变换 监听机器人在地图坐标系中的位置和姿态，并根据特定条件更新参数和调用服务。

double CarX, CarY, CarZ, CarW, CarYaw;
int teb_reload_flag = 0;

ros::ServiceClient teb_param_reload_;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "TransformListener2");

    ros::NodeHandle nh;

    // 设置 tf 的日志级别为 ERROR，忽略 WARNING 级别的日志    这个后面可以删掉，知识调试时嫌报warning麻烦
    if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Error))
    {
        ros::console::notifyLoggerLevelsChanged();
    }

    tf::TransformListener listener; // 创建一个 TF 变换监听器，用于监听坐标系之间的变换。

    listener.waitForTransform("/map", "/base_link", ros::Time(0), ros::Duration(10.0));
    // 等待 /map 坐标系到 /base_link 坐标系的变换可用，最多等待 10 秒。

    ros::Rate rate(2); // ros::Rate rate(10.0)：设置循环频率为 10Hz。
    while (nh.ok())
    {
        tf::StampedTransform transform; // 定义一个带时间戳的变换对象。
        try
        {
            listener.lookupTransform("/map", "/base_link", ros::Time(0), transform); // ：尝试获取 /map 坐标系到 /base_link 坐标系的最新变换信息。
        }
        catch (tf::TransformException &ex)
        {
            // 如果出现异常则输出错误信息并休眠 1 秒后继续循环。
            ROS_ERROR("%s", ex.what());
            ros::Duration(1.0).sleep();
            continue;
        }

        // ROS_INFO("Translation: [%f, %f, %f]", transform.getOrigin().x(), transform.getOrigin().y(), transform.getOrigin().z());
        // ROS_INFO("Rotation: [%f, %f, %f, %f]", transform.getRotation().x(), transform.getRotation().y(), transform.getRotation().z(), transform.getRotation().w());
        double roll, pitch, yaw;

        // 从变换对象中提取机器人的位置和姿态信息，包括 x, y 坐标和四元数的 z, w 分量。
        CarX = transform.getOrigin().x();
        CarY = transform.getOrigin().y();
        CarZ = transform.getRotation().z();
        CarW = transform.getRotation().w();

        // 将四元数转换为欧拉角（绕 x, y, z 轴的旋转角度），并提取偏航角 yaw。
        tf::Quaternion tf_quaternion(0, 0, CarZ, CarW);
        tf::Matrix3x3(tf_quaternion).getRPY(roll, pitch, yaw);
        CarYaw = yaw;

        // 将机器人的位置和姿态信息设置为 ROS 参数，方便其他节点使用。
        nh.setParam("CarX", CarX);
        nh.setParam("CarY", CarY);
        nh.setParam("CarZ", CarZ);
        nh.setParam("CarW", CarW);
        nh.setParam("CarYaw", CarYaw);

        // printf("CarX: %lf\n", CarX);
        // printf("CarY: %lf\n", CarY);
        // printf("CarYaw: %lf\n", CarYaw);

        rate.sleep();
    }
    return 0;
}
