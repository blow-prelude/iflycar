#!/bin/bash

#xfce4-terminal --tab --title="ucar_image" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun ucar_camera ucar_camera.py; exec bash'" &

#xfce4-terminal --tab --title="ucar_nav" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; roslaunch ucar_nav test1.launch; exec bash'" &

xfce4-terminal --title="traffic_light" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun traffic_light traditional_light_cv_ros.py; exec bash'" &

xfce4-terminal --title="image_process" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun vision_line find_way_ros; exec bash'" &

xfce4-terminal --title="vision_line_node" --command "bash -c 'source ~/ucar_ws/devel/setup.bash; rosrun vision_line vision_line_node_fixed; exec bash'" &

echo "五个独立窗口已启动（每个窗口已 source ucar_ws 环境）。"
