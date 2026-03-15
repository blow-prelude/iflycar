import rospy
import cv2
import numpy as np
import time
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from pyzbar.pyzbar import decode

class vision_zbar:
    def __init__(self):
        rospy.init_node('zbar_ros', anonymous=True)
        self.image_sub = rospy.Subscriber("/ucar_camera/image_raw", Image, self.image_callback, queue_size=1024)
        self.vision_zbar_pub = rospy.Publisher('/vision_zbar', String, queue_size=1024)
        self.vision_zbar_data = String()
        self.start_time = time.time()
        self.count = 0
        self.mtx = np.array([[408.41864465257, 0, 309.7039339414456],[0, 406.0509463667899, 272.3979542024044],[0, 0, 1]], dtype=np.float32)
        self.dist = np.array([-0.2816538317937755, 0.06677467622710392, -0.007883381249276117, 0.000701423277555337, 0], dtype=np.float32)

    def zbar_decode(self, image):
        decoded_objects = decode(image)
        for obj in decoded_objects:
            # 绘制矩形框
            (x, y, w, h) = obj.rect
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)



            barcode_data = obj.data.decode("utf-8")
            barcode_type = obj.type
            print(f"找到 {barcode_type} 条形码：{barcode_data}")


            self.pub(barcode_data)
        cv2.imshow("decode_image", image)
    

    def image_callback(self, data):
        bridge = CvBridge()
        global ros_image
        try:
            ros_image = bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
    
    def print_FPS(self):
        # 计算FPS
        self.count += 1
        end_time = time.time()
        fps = self.count / (end_time - self.start_time)
        print("FPS: ", fps)

    def pub(self, label):
        # 发布误差
        self.vision_zbar_data.data = label
        self.vision_zbar_pub.publish(self.vision_zbar_data)

ros_image = np.zeros((480, 640, 3), np.uint8)
ros_image.fill(255)

if __name__ == '__main__':
    vision_zbar = vision_zbar()
    while not rospy.is_shutdown():
        frame = ros_image
        vision_zbar.zbar_decode(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

