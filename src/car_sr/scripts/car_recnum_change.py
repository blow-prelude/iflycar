import rospy
from std_msgs.msg import Int32

"""
类号：
1:  水果
2:  蔬菜
3:  甜点

food_num:
1.apple             530
2.banana            609
3.watermelon        1086

4.peper             652
5.tomato            660
6.potato            663

7.milk              429
8.cake              404
9.coke              418
"""

old_food_num = 0
changed_food_num = 0


def received_food_num_callback(msg):
    global old_food_num
    old_food_num = msg.data
    if old_food_num > 0:
        data_change()


def data_change():
    global changed_food_num
    if old_food_num == 1:
        changed_food_num = 530
    elif old_food_num == 2:
        changed_food_num = 609
    elif old_food_num == 3:
        changed_food_num = 1086
    elif old_food_num == 4:
        changed_food_num = 652
    elif old_food_num == 5:
        changed_food_num = 660
    elif old_food_num == 6:
        changed_food_num = 663
    elif old_food_num == 7:
        changed_food_num = 429
    elif old_food_num == 8:
        changed_food_num = 404
    elif old_food_num == 9:
        changed_food_num = 418

    if changed_food_num > 100:
        changed_food_num_pub.publish(changed_food_num)
        rospy.loginfo("Changed food number: %d", changed_food_num)
        rate = rospy.Rate(10)
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node("car_recnum_change_node")
    print("car_recnum_change_node succeed")
    food_change_sub = rospy.Subscriber(
        "received_food_num", Int32, received_food_num_callback
    )
    changed_food_num_pub = rospy.Publisher("changed_food_num", Int32, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    rospy.spin()
