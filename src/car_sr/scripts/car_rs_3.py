import socket
import struct
import rospy
from std_msgs.msg import Int32

# TCP 配置
HOST = '192.168.1.110'  # 车端 IP
PORT = 8888

'''
类号：
1:  水果
2:  蔬菜
3:  甜点

food_num:
1.apple
2.banana
3.watermelon

4.peper
5.tomato
6.potato

7.milk
8.cake
9.coke
'''

received_command = 0

def command_callback(msg):
    global received_command
    received_command = msg.data
    print("rec",received_command)

def tcp_server():
    global received_command
    # ROS 初始化
    rospy.init_node('car_rs_node')
    rospy.Subscriber('class_command', Int32, command_callback)
    room_pub = rospy.Publisher('received_room_num', Int32, queue_size=10)
    food_pub = rospy.Publisher('received_food_num', Int32, queue_size=10)
    price_pub = rospy.Publisher('received_price', Int32, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"车端等待连接 {HOST}:{PORT}...")
        
        conn, addr = s.accept()
        with conn:
            print(f"电脑端已连接: {addr}")
            
            # 等待从ROS话题收到类号
            while not rospy.is_shutdown():
                if received_command > 0:
                    # 1. 车端发送类号给电脑端
                    conn.sendall(struct.pack('!i', received_command))
                    print("已发送 int32(类号) 给电脑端")
                    print(received_command)
                    received_command = 0  # 重置
                    break
                rospy.sleep(0.1)
            
            # 2. 等待接收 room_num 和 food_num（8 字节）
            data = conn.recv(12)
            if data:
                room_num, food_num, price = struct.unpack('!iii', data)
                print(f"收到 room_num={room_num}, food_num={food_num}, price={price}")
                
                start_time = rospy.Time.now()
                # 通过 ROS 发布收到的数据
                
                while(rospy.Time.now() - start_time < rospy.Duration(300)):
                    room_pub.publish(Int32(room_num))
                    food_pub.publish(Int32(food_num))
                    price_pub.publish(Int32(price))
                    print("已通过 ROS 发布数据")
                    rate.sleep()
                

if __name__ == '__main__':
    try:
        tcp_server()
    except rospy.ROSInterruptException:
        pass
