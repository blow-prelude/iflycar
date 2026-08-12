#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import socket
import time

import rospy


LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 9000
REMOTE_IP = "192.168.10.217"
REMOTE_PORT = 9000

SEND_COUNT = 10
RECEIVE_POLL_TIMEOUT = 1.0
SUCCESS_TIMEOUT = 600.0
MAX_DATAGRAM_SIZE = 65535


def sim_class_to_category(sim_class):
    text = str(sim_class)

    if "食品" in text or "食" in text:
        return "food"

    if "日用品" in text or "用品" in text or "日用" in text:
        return "daily"

    if "电子" in text or "电" in text or "生产" in text:
        return "electronics"

    return None


def encode_json(obj):
    # 保持与 TCP 版本一致的 UTF-8 JSON 内容；换行在 UDP 中不是分包依据。
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def send_json_repeated(sock, obj, address, count=SEND_COUNT):
    data = encode_json(obj)
    for _ in range(count):
        sock.sendto(data, address)


def recv_json(sock):
    data, address = sock.recvfrom(MAX_DATAGRAM_SIZE)
    return json.loads(data.decode("utf-8").strip()), address


def wait_remote_gazebo_success(sock, timeout=SUCCESS_TIMEOUT):
    """等待指定 ROS 主机返回 gazebo_success=1。"""
    deadline = time.monotonic() + timeout
    sock.settimeout(RECEIVE_POLL_TIMEOUT)

    while not rospy.is_shutdown() and time.monotonic() < deadline:
        try:
            reply, address = recv_json(sock)
        except socket.timeout:
            rospy.loginfo_throttle(2.0, "Waiting UDP gazebo_success message...")
            continue
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            rospy.logwarn("Ignoring invalid UDP JSON: %s", exc)
            continue

        if address[0] != REMOTE_IP:
            rospy.logwarn("Ignoring UDP message from unexpected host %s:%d", *address)
            continue

        rospy.loginfo("Received from ROS host %s:%d: %s", address[0], address[1], reply)

        if reply.get("type") != "gazebo_success":
            rospy.logwarn("Ignoring unknown UDP message: %s", reply)
            continue

        try:
            value = int(reply.get("value", 0))
        except (TypeError, ValueError):
            rospy.logwarn("Ignoring invalid gazebo_success value: %r", reply.get("value"))
            continue

        if value == 1:
            return True

        rospy.loginfo("Remote gazebo_success=0, keep waiting...")

    return False


def create_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((LISTEN_IP, LISTEN_PORT))
    rospy.loginfo("Car UDP server listening on %s:%d", LISTEN_IP, LISTEN_PORT)
    return server


def main():
    rospy.init_node("sim_bridge_car_udp_node")
    rospy.set_param("gazebo_sim_done", 0)

    server = create_server()
    remote_address = (REMOTE_IP, REMOTE_PORT)
    last_start_sent = False
    rate = rospy.Rate(5)

    rospy.loginfo(
        "sim_bridge_car_udp_node started. UDP server mode; remote=%s:%d",
        REMOTE_IP,
        REMOTE_PORT,
    )

    try:
        while not rospy.is_shutdown():
            start_flag = rospy.get_param("start_gazebo_sim", 0)

            if start_flag == 1 and not last_start_sent:
                sim_class = rospy.get_param("sim_class", "UNKNOWN")
                sim_item = rospy.get_param("sim_item", "UNKNOWN")
                sim_room = rospy.get_param("sim_room", "UNKNOWN")
                category = sim_class_to_category(sim_class)

                rospy.loginfo("Detected local start_gazebo_sim=1")
                rospy.loginfo("Simulation task from car:")
                rospy.loginfo("  sim_item  = %s", sim_item)
                rospy.loginfo("  sim_class = %s", sim_class)
                rospy.loginfo("  sim_room  = %s", sim_room)
                rospy.loginfo("  category  = %s", category)

                if category is None:
                    rospy.logerr(
                        "Cannot convert sim_class=%s to food/daily/electronics",
                        sim_class,
                    )
                    rospy.set_param("gazebo_sim_done", 0)
                    rate.sleep()
                    continue

                rospy.set_param("gazebo_sim_done", 0)
                message = {"type": "cube_category", "value": category}

                try:
                    rospy.loginfo(
                        "Sending to ROS host %s:%d (%d times): %s",
                        REMOTE_IP,
                        REMOTE_PORT,
                        SEND_COUNT,
                        message,
                    )
                    send_json_repeated(server, message, remote_address)
                    last_start_sent = True

                    success = wait_remote_gazebo_success(
                        server, timeout=SUCCESS_TIMEOUT
                    )
                    if success:
                        rospy.loginfo(
                            "Remote gazebo_success=1. Set local gazebo_sim_done=1."
                        )
                        rospy.set_param("gazebo_sim_done", 1)
                    else:
                        rospy.logerr("Remote gazebo_success timeout.")
                        rospy.set_param("gazebo_sim_done", 0)

                    rospy.set_param("start_gazebo_sim", 0)
                except OSError as exc:
                    rospy.logwarn("UDP communication error: %s", exc)
                    rospy.set_param("gazebo_sim_done", 0)
                    rospy.set_param("start_gazebo_sim", 0)

            if start_flag == 0:
                last_start_sent = False

            rate.sleep()
    finally:
        server.close()


if __name__ == "__main__":
    main()
