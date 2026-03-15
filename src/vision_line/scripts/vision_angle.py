import rospy
import cv2
import numpy as np
import time
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError

cap = cv2.VideoCapture("left.avi")
# cap = cv2.VideoCapture(0)

mtx = np.array([[404.12480204, 0, 310.87083721],
                [0, 403.5972979, 238.44523727],
                [0, 0, 1]])
dist = np.array([[-0.31301515, 0.13279955, -0.00065782, 0.00102601, -0.03491183]])


class cv_process:
    def __init__(self):
        rospy.init_node('cv_angle', anonymous=True)
        self.image_sub = rospy.Subscriber("/ucar_camera/image_raw", Image, self.image_callback, queue_size=1024)
        self.vision_angle_pub = rospy.Publisher('/vision_angle', Float32MultiArray, queue_size=1024)
        self.frame = None
        self.process_frame = None
        self.hsv_frame = None
        self.line_pairs = None
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # self.video_writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))


    def setup_mouse_callback(self, window_name):
        """设置鼠标回调函数"""
        cv2.setMouseCallback(window_name, self.on_mouse_click)

    def on_mouse_click(self, event, x, y, flags, param):
        """鼠标点击回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.hsv_frame is not None:
                hsv_value = self.hsv_frame[y, x]
                print(f"点击位置 ({x}, {y}) 的HSV值: {hsv_value}")

    def image_callback(self, data):
        bridge = CvBridge()
        global ros_image
        try:
            ros_image = bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def preprocess(self):
        # 颜色分割滤除蓝色
        self.hsv_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([50, 0, 0])
        upper_blue = np.array([179, 150, 240])
        mask = cv2.inRange(self.hsv_frame, lower_blue, upper_blue)
        cv2.imshow("mask", mask)
        mask = cv2.bitwise_not(mask)
        white = cv2.bitwise_and(self.frame, self.frame, mask=mask)
        # cv2.imshow("without_blue", frame)

        # gray
        gray = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray", gray)

        # 二值化
        ret, binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        # cv2.imshow("binary", binary)

        # 膨胀
        kernel = np.ones((5, 5), np.uint8)
        dilate = cv2.dilate(binary, kernel, iterations=1)

        dilate[:240, :] = 0

        self.process_frame = dilate

        cv2.imshow("dilate", dilate)
        # self.setup_mouse_callback("mask")
    
    def find_angle(self):
        # 轮廓检测
        lines = cv2.HoughLinesP(self.process_frame, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=5)
        line_pairs = []        
        if lines is not None:
            # 计算所有线的斜率和交点
            for i in range(len(lines)):
                for j in range(i+1, len(lines)):
                    line1 = lines[i][0]
                    line2 = lines[j][0]
                    
                    # 计算两条线的交点
                    x1, y1, x2, y2 = line1
                    x3, y3, x4, y4 = line2
                    
                    # 计算分母
                    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
                    
                    # 如果分母不为0，说明两线不平行
                    if denom != 0:
                        # 计算交点
                        ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
                        ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
                        
                        # 如果交点在两条线段范围内
                        if 0 <= ua <= 1 and 0 <= ub <= 1:
                            # 计算交点坐标
                            x = x1 + ua*(x2-x1)
                            y = y1 + ua*(y2-y1)
                            
                            # 计算每条线到交点的距离
                            dist1 = np.sqrt((x1-x)**2 + (y1-y)**2)
                            dist2 = np.sqrt((x2-x)**2 + (y2-y)**2)
                            dist3 = np.sqrt((x3-x)**2 + (y3-y)**2)
                            dist4 = np.sqrt((x4-x)**2 + (y4-y)**2)
                            
                            # 确定每条线的方向点（距离交点较远的端点）
                            p1 = (x1, y1) if dist1 > dist2 else (x2, y2)
                            p2 = (x3, y3) if dist3 > dist4 else (x4, y4)
                            
                            # 计算向量
                            vec1 = (p1[0]-x, p1[1]-y)
                            vec2 = (p2[0]-x, p2[1]-y)
                            
                            # 计算角度（0-180度）
                            angle = np.degrees(np.arccos(np.dot(vec1, vec2) / 
                                      (np.linalg.norm(vec1) * np.linalg.norm(vec2))))
                            
                            # 存储这对线、交点和角度
                            if 25 < angle < 150:
                                line_pairs.append((line1, line2, (x,y), angle))
        
        # 如果有相交的线对，选择角度最大的一对
        if line_pairs:
            # 按角度排序（从大到小）
            line_pairs.sort(key=lambda x: -x[3])
            line1, line2, intersection, angle = line_pairs[0]
            
            # 绘制这两条线
            for line in [line1, line2]:
                x1, y1, x2, y2 = line
                cv2.line(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制交点
            cv2.circle(self.frame, (int(intersection[0]), int(intersection[1])), 5, (0, 0, 255), -1)
            # 显示角度
            cv2.putText(self.frame, f"Angle: {angle:.1f}°", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            self.line_pairs = line_pairs
            return line_pairs
        
    def publish(self):
        if self.line_pairs:

            vision_angle_data = Float32MultiArray()
            vision_angle_data.data = [self.line_pairs[0][2][0], self.line_pairs[0][2][1], self.line_pairs[0][3]]
            self.vision_angle_pub.publish(vision_angle_data)

        
        
# 创建cv_process实例并设置frame
processor = cv_process()

ros_image = np.zeros((480, 640, 3), np.uint8)
ros_image.fill(255)
    
while not rospy.is_shutdown():
    # 预处理
    frame = ros_image
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.undistort(frame, mtx, dist, None, mtx)

    processor.frame = frame
        
    start_time = time.time()
    processor.preprocess()
    processor.find_angle()
    processor.publish()

    cv2.imshow("lines", processor.frame)        
    # processor.video_writer.write(processor.frame)
    # processor.setup_mouse_callback("lines")  # 添加这行

    end_time = time.time()
    time.sleep(0.01)

    # print("FPS:", 1 / max(0.001, end_time - start_time))
    if cv2.waitKey(1) & 0xFF == 27:
        processor.video_writer.release()
        break

cv2.destroyAllWindows()
cap.release()
