import rospy
import cv2
import numpy as np
import time
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError


mtx = np.array([[404.12480204, 0, 310.87083721],
                [0, 403.5972979, 238.44523727],
                [0, 0, 1]])
dist = np.array([[-0.31301515, 0.13279955, -0.00065782, 0.00102601, -0.03491183]])

# 定义y坐标的有效范围（仅发布此范围内的点）
Y_LOWER_BOUND = 340
Y_UPPER_BOUND = 370

class vision_line:
    def __init__(self):
        rospy.init_node('cv_line', anonymous=True)
        self.image_sub = rospy.Subscriber("/ucar_camera/image_raw", Image, self.image_callback, queue_size=1024)
        self.vision_line_pub = rospy.Publisher('/vision_line', Float32MultiArray, queue_size=1024)
        self.vision_line_data = Float32MultiArray()
        self.start_time = time.time()
        self.count = 0
    
    def preprocess(self, frame):
        # 颜色分割滤除蓝色
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([80, 0, 0])
        upper_blue = np.array([179, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.bitwise_not(mask)
        frame = cv2.bitwise_and(frame, frame, mask=mask)

        # 灰度化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 二值化
        ret, binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)

        # 膨胀
        kernel = np.ones((5, 5), np.uint8)
        dilate = cv2.dilate(binary, kernel, iterations=1)

        return dilate
    
    def find_midline(self, frame, process_frame):
        # 寻找中线
        x_mid = int(process_frame.shape[1] / 2)
        points_left = []
        points_right = []
        points_mid = []
        filtered_points_right = []
        filtered_points_left = []
        points = []
        rate_y = 5
        rate_x = 5

        # 底部边缘填充（避免边界检测失效）
        process_frame[process_frame.shape[0]-120:process_frame.shape[0], 0:10] = 255
        process_frame[process_frame.shape[0]-120:process_frame.shape[0], process_frame.shape[1] - 10:process_frame.shape[1]] = 255

        cv2.imshow("process_frame", process_frame)

        # 寻找左右边界
        for y in range(int(process_frame.shape[0] * 4 / 5), int(process_frame.shape[0] / 4), -rate_y):
            x_right_last = 999
            x_left_last = 999
            x_right = 999
            x_left = 999

            # 从中间向右寻找右边界
            for x in range(x_mid, process_frame.shape[1], rate_x):
                if x_right_last != 999:
                    if process_frame[y, x]== 255:
                        x_right = x_right_last
                        points_right.append((x_right, y))
                        break
                if process_frame[y, x]== 255:
                    x_right_last = x

            # 从中间向左寻找左边界
            for x in range(x_mid, 0, -rate_x):
                if x_left_last!= 999:
                    if process_frame[y, x]== 255:
                        x_left = x_left_last
                        points_left.append((x_left, y))
                        break
                if process_frame[y, x]== 255:
                    x_left_last = x

            # 更新中线
            if x_right_last!= 999 and x_left_last!= 999:
                x_mid = int((x_right_last + x_left_last) / 2)

        # 过滤右边界异常点
        for i in range(len(points_right)):
            if i == 0 or i == len(points_right) - 1:
                filtered_points_right.append(points_right[i])
            else:
                prev_diff = abs(points_right[i][0] - points_right[i-1][0])
                next_diff = abs(points_right[i][0] - points_right[i+1][0])
                if prev_diff < 50 and next_diff < 50:
                    filtered_points_right.append(points_right[i])

        # 过滤左边界异常点
        for i in range(len(points_left)):
            if i == 0 or i == len(points_left) - 1:
                filtered_points_left.append(points_left[i])
            else:
                prev_diff = abs(points_left[i][0] - points_left[i-1][0])
                next_diff = abs(points_left[i][0] - points_left[i+1][0])
                if prev_diff < 50 and next_diff < 50:
                    filtered_points_left.append(points_left[i])

        # 右边界插值
        for i in range(len(filtered_points_right) - 1):
            if abs(filtered_points_right[i][1] - filtered_points_right[i + 1][1]) > 2 * rate_y:
                k = (filtered_points_right[i][0] - filtered_points_right[i + 1][0]) / (filtered_points_right[i][1] - filtered_points_right[i + 1][1])
                for y in range(filtered_points_right[i][1], filtered_points_right[i + 1][1], -rate_y):
                    x = int((y - filtered_points_right[i][1]) * k + filtered_points_right[i][0])
                    filtered_points_right.append((x, y))

        # 左边界插值
        for i in range(len(filtered_points_left) - 1):
            if abs(filtered_points_left[i][1] - filtered_points_left[i + 1][1]) > 2 * rate_y:
                k = (filtered_points_left[i][0] - filtered_points_left[i + 1][0]) / (filtered_points_left[i][1] - filtered_points_left[i + 1][1])
                for y in range(filtered_points_left[i][1], filtered_points_left[i + 1][1], -rate_y):
                    x = int((y - filtered_points_left[i][1]) * k + filtered_points_left[i][0])
                    filtered_points_left.append((x, y))

        # 合并左右点，计算中线点并绘制
        y_dict = {}
        for x, y in filtered_points_left + filtered_points_right:
            if y not in y_dict:
                y_dict[y] = []
            y_dict[y].append(x)
        
        for y, x_list in y_dict.items():
            if len(x_list) == 2:
                x_mid = int((x_list[0] + x_list[1]) / 2)
                points_mid.append((x_mid, y))
                
                # 根据y坐标范围设置圆的颜色
                if Y_LOWER_BOUND <= y <= Y_UPPER_BOUND:
                    # y在有效范围，绘制红色圆（BGR格式）
                    cv2.circle(frame, (x_mid, y), 5, (0, 0, 255), -1)
                else:
                    # 其他y值，绘制蓝色圆（BGR格式）
                    cv2.circle(frame, (x_mid, y), 5, (255, 0, 0), -1)

        # 绘制y范围参考线（绿色虚线）
        cv2.line(
            frame, 
            (0, Y_LOWER_BOUND), 
            (frame.shape[1], Y_LOWER_BOUND), 
            (0, 255, 0),  # 绿色
            1, 
            cv2.LINE_AA
        )
        cv2.line(
            frame, 
            (0, Y_UPPER_BOUND), 
            (frame.shape[1], Y_UPPER_BOUND), 
            (0, 255, 0),  # 绿色
            1, 
            cv2.LINE_AA
        )

        cv2.imshow("midline", frame)
        return points_mid
    
    def calculate_error(self, frame, points):
        # 计算误差（备用函数，当前逻辑未使用）
        error = 0
        if len(points) > 0:
            mid_y = frame.shape[0] / 2
            sorted_points = sorted(points, key=lambda p: abs(p[1] - mid_y))
            closest_points = sorted_points[:5]
            if closest_points:
                cv2.circle(frame, (int(sum(p[0] for p in closest_points) / len(closest_points)), int(frame.shape[0] / 2)), 10, (255, 0, 255), -1)
                cv2.imshow("error", frame)
                error = sum(p[0] for p in closest_points) / len(closest_points) - frame.shape[1] / 2
        return error

    def image_callback(self, data):
        bridge = CvBridge()
        global ros_image
        try:
            ros_image = bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
    
    def print_FPS(self):
        # 计算FPS（备用函数）
        self.count += 1
        end_time = time.time()
        fps = self.count / (end_time - self.start_time)
        print("FPS: ", fps)

    def pub(self, points):
        global frame  # 引用主循环中的frame变量
        # 仅发布y值在270-300范围内的点
        valid_points = [p for p in points if Y_LOWER_BOUND <= p[1] <= Y_UPPER_BOUND]
        if valid_points:
            # 若存在有效点，发布最后一个有效点的误差和y值（保持原格式）
            x, y = valid_points[-1]
            self.vision_line_data.data = [x - frame.shape[1] / 2, y]
            self.vision_line_pub.publish(self.vision_line_data)
        # 无效点不发布任何数据

if __name__ == '__main__':
    vision_line = vision_line()
    ros_image = np.zeros((480, 640, 3), np.uint8)
    ros_image.fill(255)
    while not rospy.is_shutdown():
        frame = ros_image
        frame = cv2.undistort(frame, mtx, dist, None, mtx)
        process_frame = vision_line.preprocess(frame)
        points = vision_line.find_midline(frame, process_frame)
        print(points)  # 打印所有中线点（用于调试）
        vision_line.pub(points)  # 仅发布有效y范围内的点
        # vision_line.print_FPS()  # 按需启用FPS打印

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
