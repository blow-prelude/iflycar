# 加载 fly_ws 工作空间的环境变量
echo "正在加载 ucar_ws 工作空间环境变量..."
source ~/ucar_ws/devel/setup.bash

# 启动 roscore
echo "正在启动 roscore..."
xfce4-terminal --tab --title="roscore" --command "bash -c 'sudo pkill roscore; source ~/ucar_ws/devel/setup.bash; lsof -ti :8888 | xargs -r kill -9;sleep 2;roscore; rosparam delete /; exec bash'"
sleep 5


# 启动 ourgoal.launch
echo "正在启动 ourgoal ourgoal.launch..."
xfce4-terminal --tab --title="ourgoal" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; roslaunch ourgoal ourgoal.launch; exec bash'"
sleep 2

# 启动 ucar_camera
echo "正在启动 ucar_camera..."
xfce4-terminal --tab --title="ucar_camera" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun ucar_camera ucar_camera.py; exec bash'"
sleep 3

# 启动相机雷达重投影，发布 /vision_points
echo "正在启动 camera_lidar reprojection..."
xfce4-terminal --tab --title="reprojection" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; roslaunch camera_2d_lidar_calibration reprojection.launch; exec bash'"
sleep 3

# 启动 getLaserPoint，提供 /srv_getLaserPoint
echo "正在启动 getLaserPoint..."
xfce4-terminal --tab --title="getLaserPoint" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun ourgoal getLaserPoint; exec bash'"
sleep 2

# 启动 find_signal OCR/RKNN 视觉识别
echo "正在启动 find_signal rknn_ros.py..."
xfce4-terminal --tab --title="find_signal" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; cd /home/ucar/ucar_ws/src/find_signal/scripts; source ~/venv3.9/bin/activate; python3 rknn_ros.py; exec bash'"
sleep 3

# 启动 switch_test2
echo "正在启动 ourgoal switch_test2..."
xfce4-terminal --tab --title="switch_test2" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun ourgoal switch_test2; exec bash'"
sleep 3

# 启动 TransformListener2
echo "正在启动 ourgoal TransformListener2..."
xfce4-terminal --tab --title="TransformListener" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun ourgoal TransformListener2; exec bash'"
sleep 3

# 启动 AI
echo "正在启动 AI..."
xfce4-terminal --tab --title="AI" --command "bash -c 'source ~/ucar_ws/devel/setup.bash;cd /home/ucar/ucar_ws/src/ourgoal/scripts; source ~/venv3.9/bin/activate; python AI.py; exec bash'"
sleep 3

# 启动 speech_command
echo "正在启动 speech_command speech_command.launch..."
xfce4-terminal --tab --title="speech_command" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; roslaunch speech_command speech_command.launch; exec bash'"
sleep 3

# # 启动 callback
# echo "正在启动 callback..."
# xfce4-terminal --tab --title="callback" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun ourgoal callback; exec bash'"
# sleep 3

# 启动 laser2plc
echo "正在启动 laser2plc..."
xfce4-terminal --tab --title="laser2plc" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun pcl_work laser2plc; exec bash'"
sleep 3

# 启动 ultrasound
echo "正在启动 ultrasound..."
xfce4-terminal --tab --title="ultrasound" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun pcl_work ultrasound; exec bash'"
sleep 3

xfce4-terminal --title="traffic_light" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun traffic_light traditional_light_cv_ros.py; exec bash'" &

xfce4-terminal --title="image_process" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun vision_line find_way_ros; exec bash'" &

xfce4-terminal --title="vision_line_node" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun vision_line vision_line_node_fixed; exec bash'" &


echo "所有 ROS 服务已在新终端窗口中启动"
echo "每个程序运行在单独的标签页中，标题已设置为相应的程序名"


