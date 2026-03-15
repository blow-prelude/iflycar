# 加载 fly_ws 工作空间的环境变量
echo "正在加载 ucar_ws 工作空间环境变量..."
source ~/ucar_ws/devel/setup.bash

# 启动 roscore
echo "正在启动 roscore..."
xfce4-terminal --tab --title="roscore" --command "bash -c 'sudo pkill roscore; source ~/ucar_ws/devel/setup.bash; roscore; rosparam delete /; exec bash'"
sleep 1

# 启动 ucar_camera
echo "正在启动 ucar_camera..."
xfce4-terminal --tab --title="ucar_camera" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun ucar_camera ucar_camera.py; exec bash'"
sleep 3

# 启动 ourgoal.launch
echo "正在启动 ourgoal ourgoal.launch..."
xfce4-terminal --tab --title="ourgoal" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; roslaunch ourgoal ourgoal.launch; exec bash'"
sleep 2

# 启动 switch
echo "正在启动 ourgoal switch..."
xfce4-terminal --tab --title="switch" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun ourgoal switch; exec bash'"
sleep 3

# 启动 TransformListener2
echo "正在启动 ourgoal TransformListener2..."
xfce4-terminal --tab --title="TransformListener" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun ourgoal TransformListener2; exec bash'"
sleep 3

# 启动 vision_line.py
echo "正在启动 vision_line.py..."
xfce4-terminal --tab --title="vision_line" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; source ~/venv3.9/bin/activate; python /home/ucar/ucar_ws/src/vision_line/scripts/vision_line.py; exec bash'"
sleep 2

# 启动 vision_line
echo "正在启动 vision_line vision_line..."
xfce4-terminal --tab --title="vision_line" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun vision_line vision_line_node; exec bash'"
sleep 2

echo "所有 ROS 服务已在新终端窗口中启动"
echo "每个程序运行在单独的标签页中，标题已设置为相应的程序名"


