import cv2
import numpy as np
import rospy
import time
from sensor_msgs.msg import Image
from std_msgs.msg import Int32  # 改为发布整数类型
from cv_bridge import CvBridge, CvBridgeError

class ColorDetector:
    def __init__(self):
        rospy.init_node('color_detector', anonymous=True)
        # 订阅摄像头图像话题
        self.image_sub = rospy.Subscriber("/ucar_camera/image_raw", Image, self.image_callback, queue_size=1024)
        # 发布识别结果（整数类型）
        self.result_pub = rospy.Publisher('/color_result', Int32, queue_size=1024)
        self.result_data = Int32()  # 整数消息对象
        self.start_time = time.time()
        self.count = 0
        self.ros_image = None  # 存储图像的实例变量，避免全局变量问题
        
        # 相机内参（预留，未使用）
        self.mtx = np.array([[404.12480204, 0, 310.87083721],
                [0, 403.5972979, 238.44523727],
                [0, 0, 1]])
        self.dist = np.array([[-0.31301515, 0.13279955, -0.00065782, 0.00102601, -0.03491183]])

    def detect_color(self, image):
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 红色阈值范围
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        
        # 绿色阈值范围
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # 寻找红色轮廓
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 寻找绿色轮廓
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 优先识别面积最大的颜色
        max_area_red = 0
        max_area_green = 0
        
        if contours_red:
            max_contour_red = max(contours_red, key=cv2.contourArea)
            max_area_red = cv2.contourArea(max_contour_red)
            x, y, w, h = cv2.boundingRect(max_contour_red)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Red (2)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        if contours_green:
            max_contour_green = max(contours_green, key=cv2.contourArea)
            max_area_green = cv2.contourArea(max_contour_green)
            x, y, w, h = cv2.boundingRect(max_contour_green)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Green (1)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 发布面积最大的颜色（解决同时识别的冲突）
        if max_area_green > max_area_red and max_area_green > 1000:  # 增加面积阈值过滤噪声
            self.publish_result(1)
        elif max_area_red > max_area_green and max_area_red > 1000:
            self.publish_result(2)
            
        # 显示处理后的图像
        cv2.imshow("Color Detection", image)

    def image_callback(self, data):
        bridge = CvBridge()
        try:
            # 将ROS图像转换为OpenCV格式并存储到实例变量
            self.ros_image = bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

    def print_FPS(self):
        self.count += 1
        end_time = time.time()
        fps = self.count / (end_time - self.start_time)
        rospy.loginfo("FPS: %.2f", fps)

    def publish_result(self, value):
        self.result_data.data = value
        self.result_pub.publish(self.result_data)

if __name__ == '__main__':
    try:
        detector = ColorDetector()
        rate = rospy.Rate(30)  # 30Hz循环频率
        while not rospy.is_shutdown():
            # 确保图像已获取
            if detector.ros_image is not None:
                detector.detect_color(detector.ros_image)
                detector.print_FPS()  # 打印帧率
            
            # 处理按键事件
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
            rate.sleep()
        
        cv2.destroyAllWindows()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
    
