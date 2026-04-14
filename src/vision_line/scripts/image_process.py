import logging

import cv2
import numpy as np
import rospy
from camera_capture import CameraCapture
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

transformation_matrix = np.array(
    [
        [-0.498345, -1.637252, 251.246087],
        [-0.021773, 0.349689, -81.638093],
        [-0.000241, -0.009225, 1.000000],
    ]
)

CORNER_ANGLE_THRESH_DEG = 30
CORNER_STRIDE_K = 3
CORNER_CLUSTER_DX = 10
CORNER_CLUSTER_DY = 5
HORIZ_MIN_RUN_LEN = 100


class ImageProcess:
    def __init__(self, img_path=None, use_camera=False):
        """
        初始化图像处理对象

        Args:
            img_path: 图片路径，如果提供则从图片读取
            use_camera: 是否使用摄像头，默认False
        """
        if use_camera:
            self.cap = CameraCapture(index=0, width=640, height=480)
            self.img_path = None
        else:
            self.cap = None
            self.img_path = img_path
        self.frame = None
        self.canvas = None
        self.left_line = []
        self.right_line = []
        self.supple_left_line = []
        self.supple_right_line = []
        self.mid_line = []
        self.fit_mid_line = []

        # 上一帧的边线信息（用于指导当前帧搜索）
        self.prev_left_line = []  # 上一帧的左边线点列表
        self.prev_right_line = []  # 上一帧的右边线点列表
        self.prev_supple_left_line = []  # 上一帧优化后的左边线
        self.prev_supple_right_line = []  # 上一帧优化后的右边线

        # 搜索配置参数
        self.search_range = 100  # 搜索范围（像素），向左/右搜索的最大距离
        self.search_offset = 30  # 搜索偏移量（像素）

    def preprocess(self):
        """做预处理，得到二值化的图像"""
        try:
            # 如果frame已经设置好（视频处理模式），直接使用
            if self.frame is None:
                if self.cap is not None:
                    # 使用摄像头
                    self.frame = self.cap.get_picture()
                elif self.img_path is not None:
                    # 从图片文件读取
                    self.frame = cv2.imread(self.img_path)
                else:
                    # 既没有摄像头也没有图片路径
                    logging.error("No image source available")
                    return None

            # 检查frame是否有效
            if self.frame is None:
                logging.error("Failed to get frame")
                return None

            # 如果图片太大，按比例缩小
            if self.frame.shape[0] >= 240 or self.frame.shape[1] >= 320:
                # 将图片按比例缩小，使宽和高都不超过640和480
                h, w = self.frame.shape[:2]
                scale = min(240 / h, 320 / w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                self.frame = cv2.resize(
                    self.frame, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
                self.canvas = self.frame.copy()

            grey = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            binary = cv2.threshold(grey, 180, 255, cv2.THRESH_BINARY)[1]
            # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            # erode = cv2.erode(binary, kernel, iterations=2)  # 用腐消除图像中较亮的区域
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # close = cv2.morphologyEx(
            #     binary, cv2.MORPH_CLOSE, kernel, iterations=3
            # )  # 用闭运算消除图像中较暗的区域
            # 做膨胀操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilate = cv2.dilate(binary, kernel, iterations=1)  # 用膨胀操作连接边线
            return dilate
        except Exception as e:
            logging.error(f"Error occurred during image processing: {e}")
            return None

    def return_frame(self):
        """获取用于绘制的画布（当前帧的副本）

        Returns:
            画布图像（frame的副本），如果frame为None则返回None
        """
        if self.frame is None:
            logging.warning("Frame is None, cannot create canvas")
            return None
        return self.frame.copy()

    def perspective_transform(self, img, matrix, output_size=None):
        """对图像进行透视变换

        Args:
            matrix: 3x3透视变换矩阵
            img: 原始图像（BGR格式）
            output_size: 输出图像的尺寸 (width, height)，如果为None则使用原图像尺寸

        Returns:
            透视变换后的图像，如果变换失败则返回None
        """
        try:
            if matrix is None:
                logging.error("Perspective transform matrix is None")
                return None

            if img is None:
                logging.error("Input image is None")
                return None

            # 如果没有指定输出尺寸，使用原图像尺寸
            if output_size is None:
                output_size = (img.shape[1], img.shape[0])  # (width, height)

            # 执行透视变换
            transformed_img = cv2.warpPerspective(
                img,
                matrix,
                output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            logging.debug(
                f"Perspective transform completed, output size: {output_size}"
            )
            return transformed_img

        except Exception as e:
            logging.error(f"Error occurred during perspective transform: {e}")
            return None

    def get_perspective_matrix(self, src_points, dst_points):
        """根据源点和目标点计算透视变换矩阵

        Args:
            src_points: 源图像中的4个点，格式为 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            dst_points: 目标图像中的4个点，格式与src_points相同

        Returns:
            3x3透视变换矩阵，如果计算失败则返回None
        """
        try:
            if len(src_points) != 4 or len(dst_points) != 4:
                logging.error(
                    "Source and destination points must contain exactly 4 points"
                )
                return None

            # 转换为numpy数组并指定数据类型
            src_pts = np.array(src_points, dtype=np.float32)
            dst_pts = np.array(dst_points, dtype=np.float32)

            # 计算透视变换矩阵
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

            logging.debug("Perspective matrix computed successfully")
            return matrix

        except Exception as e:
            logging.error(f"Error occurred while computing perspective matrix: {e}")
            return None

    def _get_search_start_point(self, y_coord, prev_line, img_width, is_left=True):
        """根据上一帧边线位置获取当前帧的搜索起点

        Args:
            y_coord: 当前搜索行的y坐标
            prev_line: 上一帧的边线点列表
            img_width: 图像宽度
            is_left: 是否为左边线（True=左边线，False=右边线）

        Returns:
            int: 搜索起点的x坐标，如果没有上一帧信息则返回中线位置
        """
        # 如果没有上一帧信息，返回图像中线
        if len(prev_line) == 0:
            return img_width // 2

        # 在上一帧边线中查找y坐标最接近的点
        best_match = None
        min_y_diff = float("inf")

        for point in prev_line:
            x, y = point
            y_diff = abs(y - y_coord)
            if y_diff < min_y_diff:
                min_y_diff = y_diff
                best_match = point

        if best_match is not None:
            prev_x, prev_y = best_match
            # 左边线：从上一帧点的右侧偏移位置开始向左搜索
            # 右边线：从上一帧点的左侧偏移位置开始向右搜索
            if is_left:
                start_x = prev_x + self.search_offset
            else:
                start_x = prev_x - self.search_offset

            # 边界检查
            start_x = max(0, min(start_x, img_width - 1))
            return start_x

        # 如果没有找到匹配点，返回中线
        return img_width // 2

    def _update_prev_frame_lines(self):
        """在每帧处理完成后，更新上一帧的边线信息"""
        # 只有当当前帧成功检测到边线时才更新
        if len(self.left_line) > 0 and len(self.right_line) > 0:
            self.prev_left_line = self.left_line.copy()
            self.prev_right_line = self.right_line.copy()

            # 同时保存优化后的边线，用于更精确的y坐标匹配
            if len(self.supple_left_line) > 0 and len(self.supple_right_line) > 0:
                self.prev_supple_left_line = self.supple_left_line.copy()
                self.prev_supple_right_line = self.supple_right_line.copy()

    def _linear_interpolation(self, line_points):
        """对边线点进行线性插值，填充间隔过大的点

        Args:
            line_points: 原始边线点列表

        Returns:
            插值后的边线点列表
        """
        if len(line_points) < 2:
            return line_points.copy()

        supple_line = line_points.copy()
        i = 0
        while i < len(supple_line) - 1:
            x1, y1 = supple_line[i]
            x2, y2 = supple_line[i + 1]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            # 如果两点间距离过大，进行线性插值
            if dx > 10 or dy > 5:
                # 计算需要插入的点数量（每隔3-5个像素填充一个点）
                num_points = max(dx // 5, dy // 3, 2)

                # 在两点之间进行线性插值
                for t in range(1, num_points + 1):
                    ratio = t / (num_points + 1)
                    new_x = int(x1 + (x2 - x1) * ratio)
                    new_y = int(y1 + (y2 - y1) * ratio)
                    # 确保新点在两点之间
                    if min(y1, y2) < new_y < max(y1, y2):
                        supple_line.insert(i + t, (new_x, new_y))

                i += num_points + 1
            else:
                i += 1

        return supple_line

    def _fill_boundary(self, left_line, right_line, img_shape):
        """将边线延伸到图像边界，防止计算中线时越界

        Args:
            left_line: 左边线点列表
            right_line: 右边线点列表
            img_shape: 图像形状 (height, width)

        Returns:
            (填充后的左边线, 填充后的右边线)
        """
        img_height, img_width = img_shape[0], img_shape[1]

        # 左边线边界填充
        if len(left_line) > 0:
            bottom_point = left_line[0]
            bottom_y = bottom_point[1]
            # 从图像底部向上延伸到第一个点
            temp_left_line = []
            for j1 in range(img_height - 1, bottom_y, -2):
                # 左边线延伸到图像左边界 (x=0)
                temp_left_line.append((0, j1))
            left_line = temp_left_line + left_line

        # 右边线边界填充
        if len(right_line) > 0:
            bottom_point = right_line[0]
            bottom_y = bottom_point[1]
            # 从图像底部向上延伸到第一个点
            temp_right_line = []
            for j1 in range(img_height - 1, bottom_y, -2):
                # 右边线延伸到图像右边界
                temp_right_line.append((img_width - 1, j1))
            right_line = temp_right_line + right_line

        return left_line, right_line

    def _detect_horiz_white_corner(
        self,
        img,
        *,
        turn_right,
        horiz_start_x,
        y_lo,
        y_hi,
        exclude_center,
        exclude_margin,
        small,
        large,
        cluster_size=3,
    ):
        """在指定 ROI 内沿行搜索水平白线角点（第一个命中即返回）。

        每行只取从 horiz_start_x 朝 turn_right 方向的第一个 255 像素点 p0。

        稳定性改进（方案2）：按相邻点分簇（cluster_size 个点为一簇），用相邻簇的代表点
        计算 |dy/dx| 作为簇斜率。
        - prev_abs_slope < small 且 curr_abs_slope > large 时，认为当前簇的第一个 p0 为角点。

        Args:
            img: 二值图（0/255）
            turn_right: True 向右搜；False 向左搜
            horiz_start_x: 每行搜索起点 x
            y_lo: ROI 下边界（包含）
            y_hi: ROI 上边界（不包含）
            exclude_center: (cx,cy) 排除矩形中心；None 表示不排除
            exclude_margin: 排除矩形半边长（x/y 同值）
            small: prev_abs_slope 阈值
            large: curr_abs_slope 阈值
            cluster_size: 每簇的点数（>=1）

        Returns:
            (x,y) 角点坐标或 None
        """
        if cluster_size < 1:
            cluster_size = 1

        prev_cluster_rep = None  # 前一簇代表点 (x,y)
        prev_abs_slope = None
        cluster_buf = []  # 当前簇点缓存

        has_exclude = exclude_center is not None
        if has_exclude:
            cx, cy = exclude_center

        for y in range(y_lo, y_hi, -1):
            if turn_right:
                x_range = range(horiz_start_x, img.shape[1])
            else:
                x_range = range(horiz_start_x, -1, -1)

            for x in x_range:
                if img[y, x] != 255:
                    continue

                p0 = (x, y)

                # 排除矩形
                if (
                    has_exclude
                    and abs(p0[0] - cx) <= exclude_margin
                    and abs(p0[1] - cy) <= exclude_margin
                ):
                    # 落入排除区，直接跳过该点，并重置当前簇。
                    cluster_buf.clear()
                    prev_abs_slope = None
                    prev_cluster_rep = None
                    break

                cluster_buf.append(p0)

                # 当前簇未满，继续下一行
                if len(cluster_buf) < cluster_size:
                    break

                # 生成当前簇代表点（取整的均值坐标）
                rep_x = int(sum(p[0] for p in cluster_buf) / len(cluster_buf))
                rep_y = int(sum(p[1] for p in cluster_buf) / len(cluster_buf))
                curr_cluster_rep = (rep_x, rep_y)

                # 为“返回当前簇第一个点”预留
                curr_cluster_first = cluster_buf[0]

                # 清空，开始收集下一簇
                cluster_buf.clear()

                if prev_cluster_rep is not None:
                    dx = curr_cluster_rep[0] - prev_cluster_rep[0]
                    dy = prev_cluster_rep[1] - curr_cluster_rep[1]

                    if dy > 0:
                        if dx == 0:
                            curr_abs_slope = float("inf")
                        else:
                            curr_abs_slope = abs(dy / dx)

                        if prev_abs_slope is not None and curr_abs_slope > large:
                            return curr_cluster_first

                        prev_abs_slope = curr_abs_slope

                prev_cluster_rep = curr_cluster_rep
                break  # 每行只取第一个白像素

        return None

    def get_side_line_task_1(self, img, canvas, is_draw=False, turn_right=True):
        """从图像的中线往两边搜索，获取赛道边线

        Args:
            img: 输入的二值化图像
            canvas: 用于绘制的画布图像
            is_draw: 是否在canvas上绘制调试信息，默认为False
            turn_right: 水平白线搜索方向，True向右搜，False向左搜，默认True
        """
        mid_x = img.shape[1] // 2
        # 从图像下方（靠近车辆）开始搜索
        up_ratio = 0.55
        down_ratio = 0.90

        # --- On-the-fly corner detection state (performance) ---
        left_corner = None
        right_corner = None
        left_prev = None
        right_prev = None
        left_k1 = None
        right_k1 = None

        try:
            # 清空列表
            self.left_line.clear()
            self.right_line.clear()
            self.supple_left_line.clear()
            self.supple_right_line.clear()
            self.mid_line.clear()
            self.fit_mid_line.clear()

            # 确定使用哪组上一帧边线数据（优先使用优化后的边线）
            use_prev_supple = (
                len(self.prev_supple_left_line) > 0
                and len(self.prev_supple_right_line) > 0
            )
            prev_left = (
                self.prev_supple_left_line if use_prev_supple else self.prev_left_line
            )
            prev_right = (
                self.prev_supple_right_line if use_prev_supple else self.prev_right_line
            )

            # 第一帧或边线丢失标志
            is_first_frame = len(prev_left) == 0 or len(prev_right) == 0

            if is_first_frame:
                logging.info(
                    "First frame or no previous frame data, using center line search"
                )

            for j in range(
                int(img.shape[0] * down_ratio), int(img.shape[0] * up_ratio), -1
            ):
                # 左侧赛道线
                # 获取搜索起点
                left_start_x = self._get_search_start_point(
                    j, prev_left, img.shape[1], is_left=True
                )
                # 如果是第一帧，从中线开始搜索；否则从上一帧边线点右侧开始搜索
                search_start = mid_x if is_first_frame else left_start_x

                # 绘制左边线搜索起点
                if is_draw:
                    cv2.circle(canvas, (search_start, j), 3, (255, 0, 255), -1)

                # 计算搜索终点（避免搜索超出范围）
                search_end_left = max(0, search_start - self.search_range)

                found_left = False
                for i in range(search_start, search_end_left, -1):
                    if i <= 1:
                        break
                    if img[j, i] == 0 and img[j, i - 1] != 0:
                        logging.debug(
                            f"find left line at {i}, {j} , value : {img[j, i]}"
                        )
                        # 如果列表为空，则直接添加从黑到白的跳变点；否则要判断是否和上一个边界点连续
                        if len(self.left_line) == 0:
                            self.left_line.append((i, j))
                            left_prev = (i, j)
                        else:
                            if (
                                abs(self.left_line[len(self.left_line) - 1][0] - i) < 50
                                and abs(self.left_line[len(self.left_line) - 1][1] - j)
                                < 50
                            ):
                                self.left_line.append((i, j))
                                # 在线角点检测：p2(left_prev_prev)->p1(left_prev)->p0(i,j)
                                if left_corner is None and left_prev is not None:
                                    p0 = (i, j)
                                    dx = p0[0] - left_prev[0]
                                    dy = left_prev[1] - p0[1]  # 期望正值
                                    if dy > 0:
                                        k0 = float(np.arctan2(dx, dy))
                                        if left_k1 is not None:
                                            d = (k0 - left_k1 + np.pi) % (
                                                2 * np.pi
                                            ) - np.pi
                                            if abs(d) > np.deg2rad(
                                                CORNER_ANGLE_THRESH_DEG
                                            ):
                                                left_corner = left_prev
                                        left_k1 = k0
                                    left_prev = p0
                                found_left = True
                                break  # 找到边线后break
                            else:
                                logging.debug(
                                    f"jump too far at {i}, {j} , last point : {self.left_line[len(self.left_line) - 1]}"
                                )

                if not found_left and not is_first_frame:
                    logging.debug(f"Left line not found at row {j} within search range")

                # 右侧赛道线
                # 获取搜索起点
                right_start_x = self._get_search_start_point(
                    j, prev_right, img.shape[1], is_left=False
                )
                # 如果是第一帧，从中线开始搜索；否则从上一帧边线点左侧开始搜索
                search_start = mid_x if is_first_frame else right_start_x

                # 绘制右边线搜索起点（青色）
                if is_draw:
                    cv2.circle(canvas, (search_start, j), 3, (255, 0, 255), -1)

                # 计算搜索终点（避免搜索超出范围）
                search_end_right = min(
                    img.shape[1] - 1, search_start + self.search_range
                )

                found_right = False
                for i in range(search_start, search_end_right, 1):
                    if i >= img.shape[1] - 1:
                        break
                    if img[j, i] == 0 and img[j, i + 1] != 0:
                        logging.debug(
                            f"find right line at {i}, {j} , value : {img[j, i]}"
                        )
                        if len(self.right_line) == 0:
                            self.right_line.append((i, j))
                            right_prev = (i, j)
                        else:
                            if (
                                abs(self.right_line[len(self.right_line) - 1][0] - i)
                                < 50
                                and abs(
                                    self.right_line[len(self.right_line) - 1][1] - j
                                )
                                < 50
                            ):
                                self.right_line.append((i, j))
                                # 在线角点检测
                                if right_corner is None and right_prev is not None:
                                    p0 = (i, j)
                                    dx = p0[0] - right_prev[0]
                                    dy = right_prev[1] - p0[1]
                                    if dy > 0:
                                        k0 = float(np.arctan2(dx, dy))
                                        if right_k1 is not None:
                                            d = (k0 - right_k1 + np.pi) % (
                                                2 * np.pi
                                            ) - np.pi
                                            if abs(d) > np.deg2rad(
                                                CORNER_ANGLE_THRESH_DEG
                                            ):
                                                right_corner = right_prev
                                        right_k1 = k0
                                    right_prev = p0
                                found_right = True
                                break
                            else:
                                logging.debug(
                                    f"jump too far at {i}, {j} , last point : {self.right_line[len(self.right_line) - 1]}"
                                )

                if not found_right and not is_first_frame:
                    logging.debug(
                        f"Right line not found at row {j} within search range"
                    )

                if len(self.left_line) > 0 and len(self.right_line) > 0:
                    # 线性插值
                    self.supple_left_line = self._linear_interpolation(self.left_line)
                    self.supple_right_line = self._linear_interpolation(self.right_line)

                    # 填充边界，使线段一直延伸到左右下角
                    self.supple_left_line, self.supple_right_line = self._fill_boundary(
                        self.supple_left_line, self.supple_right_line, img.shape
                    )

            # --- 角点绘制与日志 ---
            if is_draw and canvas is not None:
                if left_corner is not None:
                    cv2.circle(canvas, left_corner, 6, (0, 255, 0), -1)
                if right_corner is not None:
                    cv2.circle(canvas, right_corner, 6, (0, 255, 0), -1)

            logging.info("[corner] left=%s right=%s", left_corner, right_corner)

            # --- 水平白线角点检测 ---
            # 仅当左右边线角点都出现在图像下方（y 归一化 > 0.75）时才开始搜索
            corner_y_thresh = int(img.shape[0] * 0.65)

            horiz_corner = None
            exclude_center = None

            if (
                left_corner is not None
                and right_corner is not None
                and (
                    left_corner[1] >= corner_y_thresh
                    or right_corner[1] >= corner_y_thresh
                )
            ):
                HORIZ_UP_RATIO = 0.35
                HORIZ_DOWN_RATIO = 0.60
                y_lo = int(img.shape[0] * HORIZ_DOWN_RATIO)
                y_hi = int(img.shape[0] * HORIZ_UP_RATIO)

                # 排除矩形：以左右边线角点作为中心（若角点缺失则不排除）
                EXCLUDE_MARGIN = 15

                if turn_right:
                    exclude_center = right_corner
                else:
                    exclude_center = left_corner

                # 搜索起点：中线末端的 x 坐标
                if len(self.mid_line) > 0:
                    horiz_start_x = self.mid_line[-1][0]
                else:
                    horiz_start_x = mid_x

                horiz_corner = self._detect_horiz_white_corner(
                    img,
                    turn_right=turn_right,
                    horiz_start_x=horiz_start_x,
                    y_lo=y_lo,
                    y_hi=y_hi,
                    exclude_center=exclude_center,
                    exclude_margin=EXCLUDE_MARGIN,
                    small=5,
                    large=10,
                    cluster_size=2,
                )

                if is_draw and canvas is not None:
                    if horiz_corner is not None:
                        cv2.circle(canvas, horiz_corner, 6, (0, 255, 255), -1)

            logging.info(
                "[horiz]corner=%s exclude_center=%s",
                horiz_corner,
                exclude_center,
            )

            # 使用优化后的边线计算中线
            for j in range(
                min(len(self.supple_left_line), len(self.supple_right_line))
            ):
                mid_x = (
                    self.supple_left_line[j][0] + self.supple_right_line[j][0]
                ) // 2
                # 使用边线点的实际y坐标，而不是循环索引
                mid_y = self.supple_left_line[j][1]
                self.mid_line.append((mid_x, mid_y))

        except Exception as e:
            logging.error(f"Error occurred during getting side lines : {e}")
        finally:
            # 更新上一帧的边线信息
            self._update_prev_frame_lines()

    def fit_polynomial(self):
        """根据self.mid_line的原始值，用二次函数拟合曲线，返回曲线上的点的列表 self.fit_mid_line"""
        self.fit_mid_line = []
        if len(self.mid_line) < 3:
            logging.warning("Not enough points for polynomial fitting")
            return

        try:
            # 提取y和x坐标（y为自变量，x为因变量）
            y_points = np.array([point[1] for point in self.mid_line])
            x_points = np.array([point[0] for point in self.mid_line])

            # 二次多项式拟合: x = a*y^2 + b*y + c
            coefficients = np.polyfit(y_points, x_points, 2)

            # 生成y的序列（从最小y到最大y）
            y_min = int(y_points.min())
            y_max = int(y_points.max())

            # 计算拟合曲线上的点
            for y in range(y_min, y_max + 1):
                x = int(np.polyval(coefficients, y))
                self.fit_mid_line.append((x, y))

            logging.debug(
                f"Polynomial fitting completed: {len(self.fit_mid_line)} points, "
                f"coefficients: {coefficients}"
            )
        except Exception as e:
            logging.error(f"Error occurred during polynomial fitting: {e}")

    def draw_line(self, canvas):
        """绘制边线、中线

        Args:
            canvas: 用于绘制的画布图像

        Returns:
            绘制后的画布图像，如果出错则返回None
        """
        try:
            if canvas is None:
                logging.error("Canvas is None")
                return None

            # 检查并转换为3通道BGR格式
            if len(canvas.shape) == 2:
                # 单通道图像（灰度图），转换为BGR
                canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
                logging.debug("Canvas converted from grayscale to BGR")
            elif canvas.shape[2] != 3:
                # 非3通道图像，转换为BGR
                canvas = cv2.cvtColor(
                    canvas,
                    cv2.COLOR_BGRA2BGR if canvas.shape[2] == 4 else cv2.COLOR_GRAY2BGR,
                )
                logging.debug(
                    f"Canvas converted to BGR (original channels: {canvas.shape[2]})"
                )

            logging.debug(
                f"length of left_line: {len(self.supple_left_line)} , lenth of right_line: {len(self.supple_right_line)}"
            )
            # 绘制优化后的左边线（红色）
            for i in range(len(self.supple_left_line)):
                cv2.circle(canvas, self.supple_left_line[i], 2, (0, 0, 255), -1)
            # 绘制优化后的右边线（绿色）
            for i in range(len(self.supple_right_line)):
                cv2.circle(canvas, self.supple_right_line[i], 2, (0, 0, 255), -1)
            for i in range(len(self.mid_line)):
                cv2.circle(canvas, self.mid_line[i], 2, (255, 0, 0), -1)
            for i in range(len(self.fit_mid_line)):
                cv2.circle(canvas, self.fit_mid_line[i], 2, (255, 255, 255), -1)
            return canvas
        except Exception as e:
            logging.error(f"Error occurred during drawing lines: {e}")
            return None

    def plot_column_histogram(
        self,
        binary_img,
        start_row_ratio=0.0,
        end_row_ratio=1.0,
        min_peak_distance=20,
        min_peak_height=None,
    ):
        """沿x轴计算每列的白色像素点，并绘制成直方图

        Args:
            binary_img: 二值化后的图像（numpy数组）
            start_row_ratio: 起始行比例（0.0-1.0），0.0表示从图像顶部开始
            end_row_ratio: 结束行比例（0.0-1.0），1.0表示到图像底部结束
            min_peak_distance: 峰值之间的最小距离（像素），默认20
            min_peak_height: 峰值的最小高度，默认为最大值的30%

        Returns:
            hist_img: 绘制好直方图的图像
            white_counts: 每列白色像素点数量的数组
            peaks: 峰值的x坐标列表，从左到右排序
        """
        try:
            if binary_img is None:
                raise ValueError("binary_img is None")

            # 参数验证
            if not (0.0 <= start_row_ratio < end_row_ratio <= 1.0):
                raise ValueError(
                    "start_row_ratio must be less than end_row_ratio, both in range [0.0, 1.0]"
                )

            # 计算实际行范围
            img_height = binary_img.shape[0]
            start_row = int(start_row_ratio * img_height)
            end_row = int(end_row_ratio * img_height)

            # 提取指定区域的图像
            roi = binary_img[start_row:end_row, :]

            # 计算每列的白色像素点数量（只统计指定行范围）
            white_counts = np.sum(roi == 255, axis=0)

            # 创建直方图图像
            hist_height = 300
            hist_img = np.zeros((hist_height, binary_img.shape[1], 3), dtype=np.uint8)

            # 归一化到图像高度
            if white_counts.max() > 0:
                normalized_counts = (
                    white_counts / white_counts.max() * (hist_height - 20)
                ).astype(int)
            else:
                normalized_counts = white_counts

            # 绘制直方图
            for x, count in enumerate(normalized_counts):
                cv2.line(
                    hist_img,
                    (x, hist_height - 1),
                    (x, hist_height - 1 - count),
                    (255, 255, 255),
                    1,
                )

            # 检测峰值
            peaks = self._find_peaks(white_counts, min_peak_distance, min_peak_height)

            # 在直方图上标记峰值位置
            for peak_x in peaks:
                peak_height = int(normalized_counts[peak_x])
                # 绘制峰值标记线（红色竖线）
                cv2.line(
                    hist_img,
                    (peak_x, hist_height - 1),
                    (peak_x, hist_height - 1 - peak_height),
                    (0, 0, 255),
                    2,
                )
                # 在峰值顶部绘制圆点
                cv2.circle(
                    hist_img,
                    (peak_x, hist_height - 1 - peak_height),
                    5,
                    (0, 0, 255),
                    -1,
                )

            # 添加边框和标签
            cv2.rectangle(
                hist_img,
                (0, 0),
                (hist_img.shape[1] - 1, hist_img.shape[0] - 1),
                (255, 255, 255),
                2,
            )

            logging.info(
                f"Column histogram generated: row range [{start_row}:{end_row}] "
                f"({start_row_ratio * 100:.1f}%-{end_row_ratio * 100:.1f}%), "
                f"max white pixels = {white_counts.max()}, "
                f"peaks found: {len(peaks)} at positions: {peaks}"
            )
            return hist_img, white_counts, peaks

        except Exception as e:
            logging.error(f"Error occurred during plotting column histogram: {e}")
            return None, None, []

    def _find_peaks(self, data, min_distance=20, min_height=None):
        """查找数据中的峰值

        Args:
            data: 输入数据数组
            min_distance: 峰值之间的最小距离
            min_height: 峰值的最小高度，默认为最大值的30%

        Returns:
            peaks: 峰值位置的列表，按x坐标排序
        """
        if min_height is None:
            min_height = data.max() * 0.3 if data.max() > 0 else 0

        peaks = []
        n = len(data)

        for i in range(n):
            # 检查当前点是否大于最小高度
            if data[i] < min_height:
                continue

            # 检查是否是局部最大值
            is_peak = True
            half_window = min_distance // 2

            # 检查左边
            left_start = max(0, i - half_window)
            if i > left_start and data[i] <= max(data[left_start:i]):
                is_peak = False

            # 检查右边
            right_end = min(n, i + half_window + 1)
            if i < right_end - 1 and data[i] <= max(data[i + 1 : right_end]):
                is_peak = False

            if is_peak:
                # 检查是否与已找到的峰值太近
                too_close = False
                for existing_peak in peaks:
                    if abs(i - existing_peak) < min_distance:
                        too_close = True
                        # 如果新峰值更高，替换旧峰值
                        if data[i] > data[existing_peak]:
                            peaks.remove(existing_peak)
                        else:
                            break
                if not too_close:
                    peaks.append(i)

        # 按x坐标排序
        peaks.sort()
        return peaks

    def slide_window(
        self,
        bin_img,
        left_x_base,
        right_x_base,
        nwindows=8,
        win_half_width=50,
        min_threshold=100,
    ):
        """滑动窗口法寻找赛道边线

        Args:
            white_counts: 每列白色像素点数量的数组
            window_size: 窗口大小
            min_threshold: 最小阈值，小于该值的列将被忽略

        Returns:
            left_x: 左边线x坐标
            right_x: 右边线x坐标
        """
        try:
            # 计算窗口的上下左右边界
            start_row_ratio = 0.6
            end_row_ratio = 0.9
            img_height = bin_img.shape[0]
            high_row = int(start_row_ratio * img_height)
            low_row = int(end_row_ratio * img_height)
            win_height = (low_row - high_row) // nwindows
            left_x_cur = left_x_base
            right_x_cur = right_x_base

            lwin_y_low = high_row
            lwin_y_high = lwin_y_low + win_height
            lwin_x_low = left_x_cur - win_half_width
            lwin_x_high = left_x_cur + win_half_width

            rwin_y_low = high_row
            rwin_y_high = lwin_y_low + win_height
            rwin_x_low = right_x_cur - win_half_width
            rwin_x_high = right_x_cur + win_half_width

            # 绘制窗口

            # 找到窗口中的非零像素点

            # 如果窗口中的非零像素点数量大于阈值，则更新窗口
        except Exception as e:
            logging.error(f"Error occurred during sliding window: {e}")
            return None, None


class ImageProcessRosNode:
    def __init__(self):
        rospy.init_node("image_process", anonymous=True)
        self.bridge = CvBridge()
        self.imgprocess = ImageProcess()
        self.frame_count = 0

        image_topic = rospy.get_param("~image_topic", "ucar_camera/image_raw")
        self.image_sub = rospy.Subscriber(
            image_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )
        logging.info("Subscribed image topic: %s", image_topic)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            logging.error("CvBridge conversion failed: %s", e)
            return

        self.frame_count += 1
        self.imgprocess.frame = frame

        binary_img = self.imgprocess.preprocess()
        if binary_img is None:
            logging.warning("Frame %d preprocessing failed", self.frame_count)
            return

        canvas = self.imgprocess.return_frame()
        if canvas is None:
            logging.warning("Frame %d failed to get canvas", self.frame_count)
            return

        self.imgprocess.get_side_line_task_1(binary_img, canvas, is_draw=True)
        self.imgprocess.fit_polynomial()
        canvas = self.imgprocess.draw_line(canvas)

        cv2.imshow("binary", binary_img)
        if canvas is not None:
            cv2.imshow("processed_img", canvas)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            rospy.signal_shutdown("User requested exit")

    def spin(self):
        try:
            rospy.spin()
        finally:
            cv2.destroyAllWindows()


def main():
    node = ImageProcessRosNode()
    node.spin()


if __name__ == "__main__":
    main()
