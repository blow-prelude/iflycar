#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

cv_image = None

def image_callback(msg):
    global cv_image
    try:
        bridge = CvBridge()
        # 将ROS图像消息转换为OpenCV图像
        cv_image = bridge.imgmsg_to_cv2(msg)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
    
    # 在窗口中显示图像
    # cv2.imshow("Image Window", cv_image)
    # cv2.waitKey(3)

if __name__ == '__main__':
    # 初始化ROS节点
    rospy.init_node('image_subscriber', anonymous=True)
    
    # 订阅图像话题
    rospy.Subscriber("/reprojection", Image, image_callback)
    
    # 保持节点运行，直到被关闭
    rospy.spin()
    
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
