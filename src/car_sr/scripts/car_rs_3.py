#!/usr/bin/env python3
import socket
import struct
import threading
import time

import rospy
from std_msgs.msg import Int32

# TCP 配置
HOST = "192.168.1.110"  # 车端 IP
PORT = 8888
RECONNECT_INTERVAL = 1  # 重连间隔(秒)

# 全局状态变量
tcp_socket = None
tcp_conn = None
tcp_active = False
shutdown_flag = False
received_command = 0

# ROS 发布器
room_pub = None
food_pub = None
price_pub = None


def accept_connection():
    """接受客户端连接"""
    global tcp_socket, tcp_conn, tcp_active

    try:
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tcp_socket.bind((HOST, PORT))
        tcp_socket.listen()
        rospy.loginfo(f"车端等待连接 {HOST}:{PORT}...")

        tcp_conn, addr = tcp_socket.accept()
        tcp_conn.settimeout(2)
        tcp_active = True
        rospy.loginfo(f"电脑端已连接: {addr}")
        return True
    except (socket.timeout, socket.error) as e:
        rospy.logwarn(f"连接错误: {e}")
        if tcp_conn:
            tcp_conn.close()
        if tcp_socket:
            tcp_socket.close()
        tcp_active = False
        return False


def send_command():
    """发送类号给电脑端"""
    global tcp_active, received_command

    try:
        if not tcp_active or received_command <= 0:
            return False

        tcp_conn.sendall(struct.pack("!i", received_command))
        rospy.loginfo(f"已发送类号: {received_command}")
        return True
    except (socket.timeout, socket.error) as e:
        rospy.logwarn(f"发送失败: {e}")
        tcp_active = False
        if tcp_conn:
            tcp_conn.close()
        return False


def receive_and_publish_data():
    """接收数据并通过ROS发布"""
    global tcp_active

    try:
        if not tcp_active:
            return None

        data = tcp_conn.recv(12)
        if not data:
            raise socket.error("连接已关闭")

        room_num, food_num, price = struct.unpack("!iii", data)
        rospy.loginfo(f"收到数据: room={room_num}, food={food_num}, price={price}")

        # 通过ROS发布收到的数据
        room_pub.publish(Int32(room_num))
        food_pub.publish(Int32(food_num))
        price_pub.publish(Int32(price))
        rospy.loginfo("已通过ROS发布数据")

        return (room_num, food_num, price)
    except (socket.timeout, socket.error) as e:
        rospy.logwarn(f"接收失败: {e}")
        tcp_active = False
        if tcp_conn:
            tcp_conn.close()
        return None


def tcp_main_loop():
    """TCP主运行循环"""
    global received_command, shutdown_flag

    while not shutdown_flag:
        if not tcp_active:
            if not accept_connection():
                time.sleep(RECONNECT_INTERVAL)
                continue

        # 1. 发送类号
        if received_command > 0:
            if send_command():
                received_command = 0  # 重置

        # 2. 接收数据并发布
        receive_and_publish_data()

        time.sleep(0.1)


def command_callback(msg):
    """类号订阅回调"""
    global received_command
    received_command = msg.data


def setup_ros():
    """初始化ROS组件"""
    global room_pub, food_pub, price_pub

    rospy.init_node("car_rs_node")
    room_pub = rospy.Publisher("received_room_num", Int32, queue_size=10)
    food_pub = rospy.Publisher("received_food_num", Int32, queue_size=10)
    price_pub = rospy.Publisher("received_price", Int32, queue_size=10)
    rospy.Subscriber("class_command", Int32, command_callback)


def main():
    global shutdown_flag

    setup_ros()

    # 启动TCP线程
    tcp_thread = threading.Thread(target=tcp_main_loop)
    tcp_thread.daemon = True
    tcp_thread.start()

    rospy.loginfo("车端TCP服务已启动，等待连接和数据...")

    try:
        # 主循环只负责保持节点运行
        while not rospy.is_shutdown():
            time.sleep(1)
    except rospy.ROSInterruptException:
        pass
    finally:
        shutdown_flag = True
        if tcp_conn:
            tcp_conn.close()
        if tcp_socket:
            tcp_socket.close()
        tcp_thread.join()
        rospy.loginfo("车端TCP服务已停止")


if __name__ == "__main__":
    main()
