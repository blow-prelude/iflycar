#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header


class UcarCamera:
    def __init__(self):
        self.mtx = np.array(
            [
                [420.88617453, 0.0, 322.17160714],
                [0.0, 423.22330218, 231.10564846],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )  # 内参数矩阵
        self.dist = np.array(
            [[-0.34912917, 0.17532058, 0.01055267, -0.00249659, 0.0]],
            dtype=np.float32,
        )  # 畸变系数

        rospy.init_node("ucar_camera", anonymous=True)

        self.calib_width = int(rospy.get_param("~calib_width", default=640))
        self.calib_height = int(rospy.get_param("~calib_height", default=480))
        self.enable_undistort = bool(rospy.get_param("~enable_undistort", default=True))
        self.undistort_alpha = float(rospy.get_param("~undistort_alpha", default=1.0))
        self._map_size = None
        self._mapx = None
        self._mapy = None

        self.img_width = int(rospy.get_param("~image_width", default=640))  # 1280
        self.img_height = int(rospy.get_param("~image_height", default=480))  # 720

        self.camera_topic_name = rospy.get_param(
            "~cam_topic_name", default="/ucar_camera/image_raw"
        )
        self.cam_pub = rospy.Publisher(
            self.camera_topic_name, Image, queue_size=1
        )  # 定义发布器

        self.image_temp = Image()  # 创建一个ROS的用于发布图片的Image()消息
        self.image_temp.header.frame_id = "opencv"  # 定义图片header里的id号
        self.image_temp.height = self.img_height  # 定义图片高度
        self.image_temp.width = self.img_width  # 定义图片宽度
        self.image_temp.encoding = "bgr8"  # 图片格式
        self.image_temp.is_bigendian = 0
        self.image_temp.step = (
            self.img_width * 3
        )  # 告诉ROS图片每行的大小，3是3byte像素（bgr）
        ## 设置摄像头相关信息
        device_path = rospy.get_param("device_path", default="/dev/video0")
        self.cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
        self.cap.set(3, self.img_width)
        self.cap.set(4, self.img_height)
        fourcc = rospy.get_param("~fourcc", default="")
        if isinstance(fourcc, str) and len(fourcc) == 4:
            codec = cv2.VideoWriter_fourcc(*fourcc)
            self.cap.set(cv2.CAP_PROP_FOURCC, codec)
        self.cam_pub_rate = int(rospy.get_param("~rate", default=30))
        ros_rate = rospy.Rate(self.cam_pub_rate)  # 定义发布频率
        while not rospy.is_shutdown():
            ros_rate = rospy.Rate(self.cam_pub_rate)  # 初始化发布频率
            ret, frame_1 = (
                self.cap.read()
            )  ##ret 为布尔值表示是否可以获得图像    frame为获取的帧
            if not ret or frame_1 is None:
                rospy.logwarn_throttle(1.0, "camera read failed")
                ros_rate.sleep()
                continue
            # cv2.imshow("raw_frame", frame_1)
            if self.enable_undistort:
                frame_1 = self.correct_img(frame_1)
            frame_1 = cv2.flip(frame_1, 1)
            cv2.imshow("frame", frame_1)
            cv2.waitKey(1)
            frame_h, frame_w = frame_1.shape[:2]
            self.image_temp.header = Header(stamp=rospy.Time.now())  # 定义图片header
            self.image_temp.height = frame_h
            self.image_temp.width = frame_w
            self.image_temp.encoding = "bgr8"
            self.image_temp.step = frame_w * 3
            self.image_temp.data = frame_1.tobytes()
            self.cam_pub.publish(self.image_temp)
            ros_rate.sleep()

    def correct_img(self, img):
        h, w = img.shape[:2]
        try:
            if len(self.mtx) > 0 and len(self.dist) > 0:
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                    self.mtx, self.dist, (w, h), 0, (w, h)
                )

            # 生成去畸变映射表，并应用映射表将像素重新映射
            mapx, mapy = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, None, newcameramtx, (w, h), 5
            )
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            return dst
        except Exception as e:
            rospy.logerr(f"error when correct img with mtx:{e}")
            return img


if __name__ == "__main__":
    ucar_camera = UcarCamera()
