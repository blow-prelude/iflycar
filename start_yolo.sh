# 加载 fly_ws 工作空间的环境变量
echo "正在加载 ucar_ws 工作空间环境变量..."
source ~/ucar_ws/devel/setup.bash

# 启动 roscore
echo "正在启动 roscore..."
xfce4-terminal --tab --title="roscore" --command "bash -c 'sudo pkill roscore; source ~/ucar_ws/devel/setup.bash; roscore; exec bash'"
sleep 3

# 启动 ucar_camera
echo "正在启动 ucar_camera..."
xfce4-terminal --tab --title="ucar_camera" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun ucar_camera ucar_camera.py; exec bash'"
sleep 3

# 启动 rknn_ros
echo "正在启动 rknn_ros..."
xfce4-terminal --tab --title="rknn_ros" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; source ~/venv3.9/bin/activate; python /home/ucar/ucar_ws/src/rknn_ros/scripts/rknn_ros.py; exec bash'"
sleep 3

# 启动 lidar
echo "正在启动 lidar..."
xfce4-terminal --tab --title="lidar" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; roslaunch ucar_nav test1.launch; exec bash'"
sleep 3

# 启动 camera_2d_lidar_calibration
echo "正在启动 camera_2d_lidar_calibration..."
xfce4-terminal --tab --title="camera_2d_lidar_calibration" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; roslaunch camera_2d_lidar_calibration reprojection.launch; exec bash'"
sleep 3

# 启动 getLaserPoint
echo "正在启动 getLaserPoint..."
xfce4-terminal --tab --title="getLaserPoint" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun ourgoal getLaserPoint; exec bash'"
sleep 3

# 启动 callback
echo "正在启动 callback..."
xfce4-terminal --tab --title="callback" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun ourgoal callback; exec bash'"
sleep 3



echo "所有 ROS 服务已在新终端窗口中启动"
echo "每个程序运行在单独的标签页中，标题已设置为相应的程序名"


