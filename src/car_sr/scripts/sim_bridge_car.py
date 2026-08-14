#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import json
import time
import rospy


LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 9000

RECONNECT_INTERVAL = 1.0
SUCCESS_TIMEOUT = 600.0


def sim_class_to_category(sim_class):
    text = str(sim_class)

    if "食品" in text or "食" in text:
        return "food"

    if "日用品" in text or "用品" in text or "日" in text:
        return "daily"

    if "电子" in text or "电" in text or "产品" in text:
        return "electronics"

    return None


def send_json(conn, obj):
    data = json.dumps(obj, ensure_ascii=False) + "\n"
    conn.sendall(data.encode("utf-8"))


def recv_json(conn, timeout=SUCCESS_TIMEOUT):
    conn.settimeout(timeout)

    buf = b""

    while not rospy.is_shutdown():
        ch = conn.recv(1)

        if not ch:
            raise ConnectionError("socket closed")

        if ch == b"\n":
            break

        buf += ch

    return json.loads(buf.decode("utf-8"))

def wait_vm_gazebo_success(conn, timeout=SUCCESS_TIMEOUT):
    """
    循环等待虚拟机返回 gazebo_success=1。

    虚拟机可能先返回：
        {"type":"gazebo_success","value":0}

    value=0 表示还没完成，不是失败。
    """
    start_time = time.time()

    while not rospy.is_shutdown():
        if time.time() - start_time > timeout:
            return False

        try:
            reply = recv_json(conn, timeout=5.0)
        except socket.timeout:
            rospy.loginfo_throttle(2.0, "Waiting VM gazebo_success message...")
            continue

        rospy.loginfo("Received from VM: %s", reply)

        if reply.get("type") != "gazebo_success":
            rospy.logwarn("Ignoring unknown VM message: %s", reply)
            continue

        value = int(reply.get("value", 0))

        if value == 1:
            return True

        rospy.loginfo("VM gazebo_success=0, keep waiting...")

def create_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((LISTEN_IP, LISTEN_PORT))
    server.listen(1)

    rospy.loginfo("Car TCP server listening on %s:%d", LISTEN_IP, LISTEN_PORT)

    return server


def accept_client(server):
    while not rospy.is_shutdown():
        try:
            rospy.loginfo("Waiting VM client connection...")
            conn, addr = server.accept()
            rospy.loginfo("VM client connected: %s", addr)
            conn.settimeout(None)
            return conn
        except Exception as e:
            rospy.logwarn("accept failed: %s", e)
            time.sleep(RECONNECT_INTERVAL)

    return None


def main():
    rospy.init_node("sim_bridge_car_node")

    rospy.set_param("gazebo_sim_done", 0)

    server = create_server()
    conn = None

    last_start_sent = False
    rate = rospy.Rate(5)

    rospy.loginfo("sim_bridge_car_node started. TCP SERVER mode.")

    while not rospy.is_shutdown():
        if conn is None:
            conn = accept_client(server)
            if conn is None:
                break

        start_flag = rospy.get_param("start_gazebo_sim", 0)

        try:
            if start_flag == 1 and not last_start_sent:
                rospy.loginfo("Detected local start_gazebo_sim=1")

                sim_class = rospy.get_param("sim_class", "UNKNOWN")
                sim_item = rospy.get_param("sim_item", "UNKNOWN")
                sim_room = rospy.get_param("sim_room", "UNKNOWN")

                category = sim_class_to_category(sim_class)

                rospy.loginfo("Simulation task from car:")
                rospy.loginfo("  sim_item  = %s", sim_item)
                rospy.loginfo("  sim_class = %s", sim_class)
                rospy.loginfo("  sim_room  = %s", sim_room)
                rospy.loginfo("  category  = %s", category)

                if category is None:
                    rospy.logerr("Cannot convert sim_class=%s to food/daily/electronics", sim_class)
                    rospy.set_param("gazebo_sim_done", 0)
                    last_start_sent = False
                    rate.sleep()
                    continue

                rospy.set_param("gazebo_sim_done", 0)

                # 发给虚拟机客户端的格式
                msg = {
                    "type": "cube_category",
                    "value": category
                }

                rospy.loginfo("Sending to VM: %s", msg)
                send_json(conn, msg)

                last_start_sent = True

                rospy.loginfo("Waiting gazebo_success=1 from VM...")

                success = wait_vm_gazebo_success(conn, timeout=SUCCESS_TIMEOUT)

                if success:
                    rospy.loginfo("VM gazebo_success=1. Set local gazebo_sim_done=1.")
                    rospy.set_param("gazebo_sim_done", 1)
                    rospy.set_param("start_gazebo_sim", 0)

                else:
                    rospy.logerr("VM gazebo_success timeout.")
                    rospy.set_param("gazebo_sim_done", 0)
                    rospy.set_param("start_gazebo_sim", 0)

            if start_flag == 0:
                last_start_sent = False

        except Exception as e:
            rospy.logwarn("TCP connection error: %s", e)

            try:
                conn.close()
            except Exception:
                pass

            conn = None
            last_start_sent = False
            time.sleep(RECONNECT_INTERVAL)

        rate.sleep()

    try:
        server.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()