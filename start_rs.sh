# 加载 fly_ws 工作空间的环境变量
echo "正在加载 ucar_ws 工作空间环境变量..."
source ~/ucar_ws/devel/setup.bash

# 启动 roscore
echo "正在启动 roscore..."
xfce4-terminal --tab --title="roscore" --command "bash -c 'sudo pkill roscore; source ~/ucar_ws/devel/setup.bash; lsof -ti :8888 | xargs -r kill -9;roscore; rosparam delete /; exec bash'"
sleep 2

# 启动 car_sr
echo "正在启动 car_sr..."
xfce4-terminal --tab --title="car_sr" --command "bash -c 'source ~/ucar_ws/devel/setup.bash;cd /home/ucar/ucar_ws/src/car_sr/scripts; source ~/venv3.9/bin/activate; python car_rs_3.py; exec bash'"
xfce4-terminal --tab --title="car_sr" --command "bash -c 'source ~/ucar_ws/devel/setup.bash;cd /home/ucar/ucar_ws/src/car_sr/scripts; source ~/venv3.9/bin/activate; python car_recnum_change.py; exec bash'"
sleep 3
# rostopic pub -r 10 /pc_command std_msgs/Int32 " 1"
echo "所有 ROS 服务已在新终端窗口中启动"
echo "每个程序运行在单独的标签页中，标题已设置为相应的程序名"


