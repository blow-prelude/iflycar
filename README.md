<h2 align="center" style="color: skyblue;">2026年智能车创意赛道讯飞组 ——</h2>
<h2 align="center" style="color: skyblue;">智慧工厂</h2>

### 环境
- 上位机：rk3588s，arm64位
- 操作系统：debian10
- ROS noetic
- python：ROS预装3.8， 部分节点使用虚拟环境3.9

---
### 功能包功能简介
#### camera_2d_lidar_calibration
RGB相机和二维雷达联合标定

#### car_sr
已弃用。原本用于arm开发板和局域网内其他设备的TCP通信

#### costmap_prohibition_layer
放置虚拟墙、禁止区

#### fdilink_ahrs
imu驱动

#### find_signal
用ocr_rknn推理，识别文字

#### geometry
用于tf

#### geometry2
用于tf2

#### navigation
用于导航+路径规划的第三方库

#### object_information_msgs
ocr推理收发有效数据的自定义消息包

#### ourgoal
前往目标点的业务逻辑

#### pcl_work
激光雷达数据转pcl点云数据

#### rknn
已弃用。原本是上一年的rknn推理代码

#### rknn_ros
已弃用。原本是上一年的rknn推理的ros部署

#### speech_command
麦克风阵列驱动

#### startup_scripts
已弃用。原本是一键启动脚本

#### traffic_light
用rknn推理+cv过滤，识别交通灯

#### ucar_camera
RGB相机节点

#### ucar_controller
无刷电机、LED、IMU的控制包

#### ucar_map
SLAM建图

#### ucar_nav
导航

#### vision_line
视觉巡线+控制命令

#### vision_ros
已弃用。原本是上一年的视觉巡线的ros端部署

#### ydlidar
激光雷达驱动


---
### 更新日志
[视觉巡线更新日志](./src/vision_line/README.md)