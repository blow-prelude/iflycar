import logging
import os
import time
from enum import Enum

import cv2
import numpy as np
from camera_capture import CameraCapture
from picture_cli import ImageSender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ProcessState(Enum):
    IDLE = 0  # 未收到指令，不处理
    STRAIGHT_TRACKING = 1  # 直行循迹，find_corner=False
    RIGHT_TRACKING = 5  # 右转循迹（占位）
    LEFT_TRACKING = 6  # 左转循迹（占位）
    CORNER = 2  # 延时已到，find_corner=True
    CROSS = 3  # 双拐点触发，find_corner=False，执行额外处理
    TURNING = 4  # 检测到停止线后的转弯状态


transformation_matrix = np.array(
    [
        [-0.498345, -1.637252, 251.246087],
        [-0.021773, 0.349689, -81.638093],
        [-0.000241, -0.009225, 1.000000],
    ]
)


class ImageProcess:
    def __init__(self, img_path=None, img=None):
        """
        初始化图像处理对象

        Args:
            img_path: 图片路径，如果提供则从图片读取
            use_camera: 是否使用摄像头，默认False
        """

        self.img_path = img_path
        self.frame = img
        self.canvas = None
        self.left_line = []
        self.right_line = []
        self.supple_left_line = []
        self.supple_right_line = []
        self.mid_line = []
        self.fit_mid_line = []

        self.left_c = None  # 本帧左边线拐点 (x, y)，未检测到时为 None
        self.right_c = None  # 本帧右边线拐点 (x, y)，未检测到时为 None

        # 上一帧的边线信息（用于指导当前帧搜索）
        self.prev_left_line = []  # 上一帧的左边线点列表
        self.prev_right_line = []  # 上一帧的右边线点列表
        self.prev_supple_left_line = []  # 上一帧优化后的左边线
        self.prev_supple_right_line = []  # 上一帧优化后的右边线

        self.x_continual = 15
        self.y_continual = 5

        # 搜索配置参数
        self.search_range = 100  # 搜索范围（像素），向左/右搜索的最大距离
        self.search_offset = 30  # 搜索偏移量（像素）
        self.init_stable_count = 5  # 初始连续点数阈值

    def preprocess(self):
        try:
            if self.img_path is not None:
                # 从图片文件读取
                self.frame = cv2.imread(self.img_path)

            if self.frame is not None:
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
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

                # 大尺寸高斯模糊获取背景光照分布
                # 核大小应根据图像尺寸调整，通常为图像宽度的1/5到1/3
                kernel_size = (gray.shape[1] // 5 | 1, gray.shape[0] // 5 | 1)
                background = cv2.GaussianBlur(gray, kernel_size, 0)

                # 原图减去背景，得到滤除光照后的特征
                diff = cv2.subtract(gray, background)

                # 二值化（使用Otsu自适应阈值）
                binary = cv2.threshold(
                    diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
                return close
            else:
                raise ValueError("Failed to load image for preprocessing")
        except Exception as e:
            raise RuntimeError(f"Error during preprocessing: {e}")

    def return_frame(self):
        """获取用于绘制的画布（当前帧的副本）

        Returns:
            画布图像（frame的副本），如果frame为None则返回None
        """
        if self.frame is None:
            raise ValueError("Frame is None, cannot return canvas")
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

    def get_angle_k(self, k1, k2):
        """
        已知斜率计算夹角，返回角度
        return:
            angle: 角度值，单位为度，保留两位小数
        """
        cos_theta = (1 + k1 * k2) / (np.sqrt(1 + k1**2) * np.sqrt(1 + k2**2))
        cos_theta = np.clip(cos_theta, -1, 1)  # 防止浮点误差越界
        return np.degrees(np.arccos(cos_theta))

    def get_angle_p(self, p1, p2, p3):
        """
        计算以p2为顶点的角度（p1-p2-p3）

        Args:
            p1: 点1坐标 (x, y)
            p2: 顶点坐标 (x, y)
            p3: 点3坐标 (x, y)

        Returns:
            float: 角度值，单位为度
        """
        # 向量 p2->p1 和 p2->p3
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        # 计算向量长度
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        # 使用点积计算余弦值
        cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1, 1)  # 防止浮点误差越界

        # 返回角度（度）
        return np.degrees(np.arccos(cos_theta))

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

    def _add_point_with_stable_start(
        self, line, point, stable_buf, stable, x_thresh, y_thresh
    ):
        """初始稳定点检测：前 N 个点必须全部连续才加入列表

        Args:
            line: 正式边线列表
            point: 候选点 (x, y)
            stable_buf: 稳定期临时缓冲区（由调用方维护）
            stable: 稳定标志列表 [bool]，首次达到稳定点数后置为 True
            x_thresh: x 方向连续性阈值
            y_thresh: y 方向连续性阈值

        Returns:
            bool: 该点是否已确认加入正式边线
        """
        if stable[0]:
            if (
                abs(line[-1][0] - point[0]) < x_thresh
                and abs(line[-1][1] - point[1]) < y_thresh
            ):
                line.append(point)
                return True
        else:
            if len(stable_buf) == 0:
                stable_buf.append(point)
            elif (
                abs(stable_buf[-1][0] - point[0]) < x_thresh
                and abs(stable_buf[-1][1] - point[1]) < y_thresh
            ):
                stable_buf.append(point)
                if len(stable_buf) >= self.init_stable_count:
                    line.extend(stable_buf)
                    stable_buf.clear()
                    stable[0] = True
                    return True
            else:
                stable_buf.clear()
                stable_buf.append(point)
        return False

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

        new_line = []

        for i in range(len(line_points) - 1):
            x1, y1 = line_points[i]
            x2, y2 = line_points[i + 1]

            new_line.append((x1, y1))

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            if dx > 6 or dy > 3:
                num_points = max(dx // 3, dy // 2, 2)

                # 用 linspace 替代循环（更快 + 更稳定）
                xs = np.linspace(x1, x2, num_points + 2)[1:-1]
                ys = np.linspace(y1, y2, num_points + 2)[1:-1]

                for x, y in zip(xs, ys):
                    xi, yi = int(x), int(y)
                    if min(y1, y2) < yi < max(y1, y2):
                        new_line.append((xi, yi))

        # 加最后一个点
        new_line.append(line_points[-1])

        return new_line

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
            bottom_y = left_line[0][1]
            ys = np.arange(img_height - 1, bottom_y, -2)
            xs = np.zeros_like(ys)

            temp_left_line = list(zip(xs.tolist(), ys.tolist()))
            left_line = temp_left_line + left_line

        # 右边线边界填充
        if len(right_line) > 0:
            bottom_y = right_line[0][1]

            ys = np.arange(img_height - 1, bottom_y, -2)
            xs = np.full_like(ys, img_width - 1)

            temp_right_line = list(zip(xs.tolist(), ys.tolist()))
            right_line = temp_right_line + right_line

        return left_line, right_line

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

    def judge_enter_cross_state(self, img_shape, y_ratio=0.75):
        """判断是否满足进入 CROSS 状态的条件

        Args:
            img_shape: 图像形状 (height, width)
            y_ratio: y 坐标阈值比例，默认 0.75

        Returns:
            bool: 满足条件返回 True，否则 False
        """
        if self.left_c is None or self.right_c is None:
            return False
        img_h = img_shape[0]
        return self.left_c[1] >= img_h * y_ratio or self.right_c[1] >= img_h * y_ratio

    def judge_enter_turning(self, stop_mid, img_shape, y_thresh=0.78):
        """判断是否应该进入 TURNING 状态

        Args:
            stop_mid: 停止线中点坐标 (x, y) 或 None
            img_shape: 图像形状 (h, w, ...)
            y_thresh: y 坐标阈值（归一化），默认 0.78

        Returns:
            bool: True 表示应该进入 TURNING 状态
        """
        if stop_mid is None:
            return False

        y_norm = stop_mid[1] / img_shape[0]
        return y_norm > y_thresh

    def get_stop_line(self, binary_img, is_draw=False, canvas=None):
        """在 ROI 内检测水平白线并返回其中点

        Args:
            binary_img: 二值化图像 (numpy array)
            is_draw: 是否绘制调试信息
            canvas: 绘制画布 (BGR 格式)

        Returns:
            Optional[Tuple[int, int]]: 成功返回 (mid_x, mid_y)，失败返回 None
        """
        # 输入验证
        if binary_img is None:
            raise ValueError("Input binary image is None")

        h, w = binary_img.shape[:2]

        # 计算 ROI 边界
        roi_y0 = int(h * 0.55)
        roi_y1 = int(h * 0.80)
        roi_x0 = int(w * 0.30)
        roi_x1 = int(w * 0.70)
        roi = binary_img[roi_y0 : roi_y1 + 1, roi_x0 : roi_x1 + 1]

        logging.debug(f"ROI: y=[{roi_y0}, {roi_y1}], x=[{roi_x0}, {roi_x1}]")

        # 横向形态学削弱斜线
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)

        contours = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        if len(contours) == 0:
            logging.debug("No contours found in stop line ROI")
            return None

        # 找最像“横线”的轮廓
        best = max(contours, key=lambda c: cv2.boundingRect(c)[2])  # 按宽度选

        x, y, w, h = cv2.boundingRect(best)

        if w <= 20:
            return None
        # 还原到原图像
        x += roi_x0
        y += roi_y0

        mid_x = x + w // 2
        mid_y = y + h // 2

        logging.debug(f"find stop line ,mid:({mid_x}, {mid_y}),")

        # 显示ROI区域（如果需要）
        # if is_draw:
        #     # 弹窗显示ROI区域（需要转成BGR格式才能正常显示）
        #     # roi_display = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        #     cv2.imshow("stop_line_roi", roi)

        # 绘制（如果需要）
        if is_draw and canvas is not None:
            # 画中点
            cv2.circle(canvas, (mid_x, mid_y), 4, (255, 0, 255), -1)
            # 绘制白线
            cv2.rectangle(
                canvas,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

        return (mid_x, mid_y)

    def get_side_line_task_1(self, img, canvas, is_draw=False):
        """从图像的中线往两边搜索，获取赛道边线

        Args:
            img: 输入的二值化图像
            canvas: 用于绘制的画布图像
            is_draw: 是否在canvas上绘制调试信息，默认为False
        """
        mid_x = img.shape[1] // 2
        # 从图像下方（靠近车辆）开始搜索
        up_ratio = 0.55
        down_ratio = 0.90
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

            # 稳定点缓冲区及标志
            left_stable_buf = []
            right_stable_buf = []
            left_stable = [False]
            right_stable = [False]

            diff = np.diff(img == 0, axis=1)  # 计算行内黑白跳变  右-左
            # cv2.imshow("diff", (diff != 0).astype(np.uint8) * 255)    # 显示发生跳变的地方

            for y in range(
                int(img.shape[0] * down_ratio), int(img.shape[0] * up_ratio), -1
            ):
                # 获取当前行内的跳变点
                row_diff = diff[y]
                # cv2.imshow(
                #     "diff", (diff != 0).astype(np.uint8) * 255
                # )  # 显示发生跳变的地方
                # logging.info(f"len(row_diff): {len(row_diff)}")

                # 左侧赛道线
                # 获取搜索起点
                left_start_x = self._get_search_start_point(
                    y, prev_left, img.shape[1], is_left=True
                )
                # 如果是第一帧，从中线开始搜索；否则从上一帧边线点右侧开始搜索
                search_start = mid_x if is_first_frame else left_start_x

                # 绘制左边线搜索起点（紫色）
                if is_draw:
                    cv2.circle(canvas, (search_start, y), 3, (255, 0, 255), -1)

                # 计算搜索终点（避免搜索超出范围）
                search_end_left = max(0, search_start - self.search_range)

                # 左边：从白到黑，跳变为1
                candidates = np.where(row_diff[search_start:search_end_left] == 1)[0]

                if len(candidates) > 0:
                    x = candidates[-1]

                    _ = self._add_point_with_stable_start(
                        self.left_line,
                        (x, y),
                        left_stable_buf,
                        left_stable,
                        self.x_continual,
                        self.y_continual,
                    )

                # 右侧赛道线
                # 获取搜索起点
                right_start_x = self._get_search_start_point(
                    y, prev_right, img.shape[1], is_left=False
                )
                # 如果是第一帧，从中线开始搜索；否则从上一帧边线点左侧开始搜索
                search_start = mid_x if is_first_frame else right_start_x

                # 绘制右边线搜索起点（青色）
                if is_draw:
                    cv2.circle(canvas, (search_start, y), 3, (255, 255, 0), -1)

                # 计算搜索终点（避免搜索超出范围）
                search_end_right = min(
                    img.shape[1] - 1, search_start + self.search_range
                )

                candidates = np.where(row_diff[search_start:search_end_right] == 1)[0]

                if len(candidates) > 0:
                    x = candidates[0]

                    self._add_point_with_stable_start(
                        self.right_line,
                        (x, y),
                        right_stable_buf,
                        right_stable,
                        self.x_continual,
                        self.y_continual,
                    )

                if len(self.left_line) > 0 and len(self.right_line) > 0:
                    # 线性插值
                    self.supple_left_line = self._linear_interpolation(self.left_line)
                    self.supple_right_line = self._linear_interpolation(self.right_line)

                    # 填充边界，使线段一直延伸到左右下角
                    self.supple_left_line, self.supple_right_line = self._fill_boundary(
                        self.supple_left_line, self.supple_right_line, img.shape
                    )

            # 使用优化后的边线计算中线
            for y in range(
                min(len(self.supple_left_line), len(self.supple_right_line))
            ):
                mid_x = (
                    self.supple_left_line[y][0] + self.supple_right_line[y][0]
                ) // 2
                # 使用边线点的实际y坐标，而不是循环索引
                mid_y = self.supple_left_line[y][1]
                self.mid_line.append((mid_x, mid_y))

        except Exception as e:
            logging.error(f"Error occurred during getting side lines : {e}")
        finally:
            # 更新上一帧的边线信息
            self._update_prev_frame_lines()

    def get_side_line_task_2(self, img, canvas, is_draw=False, find_corner=False):
        """从图像的中线往两边搜索，获取赛道边线

        Args:
            img: 输入的二值化图像
            canvas: 用于绘制的画布图像
            is_draw: 是否在canvas上绘制调试信息，默认为False
            find_corner: 是否搜寻拐点
        """
        # 从图像中间向两边搜索，获取边线
        up_ratio = 0.55
        down_ratio = 0.95

        angle_high_thresh = 135
        angle_low_thresh = 45

        left_nxt_p, left_cur_p, left_pre_p = None, None, None
        right_nxt_p, right_cur_p, right_pre_p = None, None, None

        find_left_corner, find_right_corner = False, False
        mid_x = int(img.shape[1] // 2)

        try:
            # 清空之前搜索到的赛道线
            self.left_line.clear()
            self.right_line.clear()
            self.supple_left_line.clear()
            self.supple_right_line.clear()
            self.mid_line.clear()
            self.fit_mid_line.clear()
            self.left_c = None
            self.right_c = None

            # 稳定点缓冲区及标志
            left_stable_buf = []
            right_stable_buf = []
            left_stable = [False]
            right_stable = [False]

            diff = np.diff(img == 0, axis=1)  # 计算行内黑白跳变  右-左
            cv2.imshow("diff", (diff != 0).astype(np.uint8) * 255)  # 显示发生跳变的地方

            # 从图像下方（靠近车辆）开始搜索
            for y in range(
                int(img.shape[0] * down_ratio), int(img.shape[0] * up_ratio), -1
            ):
                row_diff = diff[y]
                # 左边：从白到黑，跳变为1
                candidates = np.where(row_diff[:mid_x] == 1)[0]

                if len(candidates) > 0:
                    x = candidates[-1]

                    # 先进行稳定点检测
                    added = self._add_point_with_stable_start(
                        self.left_line,
                        (x, y),
                        left_stable_buf,
                        left_stable,
                        self.x_continual,
                        self.y_continual,
                    )

                    # 只有稳定点才参与拐点检测
                    if added and find_corner and not find_left_corner:
                        left_nxt_p = (x, y)
                        if left_cur_p is not None and left_pre_p is not None:
                            angle = self.get_angle_p(left_nxt_p, left_cur_p, left_pre_p)
                            logging.debug(
                                f"left line angle: {angle} , pre_p: {left_pre_p},  cur_p: {left_cur_p} , nxt_p: {left_nxt_p}"
                            )
                            if angle_low_thresh < angle < angle_high_thresh:
                                logging.debug(
                                    f"slope mutation , angle: {angle} ,pre_p:{left_pre_p} , cur_p: {left_cur_p} , nxt_p: {left_nxt_p} "
                                )
                                self.left_c = left_cur_p
                                find_left_corner = True
                        # 更新点
                        left_pre_p = left_cur_p
                        left_cur_p = left_nxt_p

                # 右线
                candidates = np.where(row_diff[mid_x:] == 1)[0]
                if len(candidates) > 0:
                    x = candidates[0] + mid_x

                    # 先进行稳定点检测
                    added = self._add_point_with_stable_start(
                        self.right_line,
                        (x, y),
                        right_stable_buf,
                        right_stable,
                        self.x_continual,
                        self.y_continual,
                    )

                    # 只有稳定点才参与拐点检测
                    if added and find_corner and find_right_corner is False:
                        right_nxt_p = (x, y)
                        if right_cur_p is not None and right_pre_p is not None:
                            angle = self.get_angle_p(
                                right_nxt_p, right_cur_p, right_pre_p
                            )
                            logging.debug(
                                f"right line angle: {angle} , pre_p: {right_pre_p},  cur_p: {right_cur_p} , nxt_p: {right_nxt_p}"
                            )
                            if angle_low_thresh < angle < angle_high_thresh:
                                logging.debug(
                                    f"slope mutation , angle: {angle} ,pre_p:{right_pre_p} cur_p: {right_cur_p} , nxt_p: {right_nxt_p} "
                                )
                                self.right_c = right_cur_p
                                find_right_corner = True
                        # 更新点
                        right_pre_p = right_cur_p
                        right_cur_p = right_nxt_p

            # 线性补插，优化边线
            if len(self.left_line) > 0 and len(self.right_line) > 0:
                # 线性插值
                self.supple_left_line = self._linear_interpolation(self.left_line)
                self.supple_right_line = self._linear_interpolation(self.right_line)

                # 填充边界，使线段一直延伸到左右下角
                self.supple_left_line, self.supple_right_line = self._fill_boundary(
                    self.supple_left_line, self.supple_right_line, img.shape
                )

            # 用优化后的边线计算中线
            for j in range(
                min(len(self.supple_left_line), len(self.supple_right_line))
            ):
                line_mid_x = (
                    self.supple_left_line[j][0] + self.supple_right_line[j][0]
                ) // 2
                line_mid_y = self.supple_left_line[j][1]
                self.mid_line.append((line_mid_x, line_mid_y))

            # 多项式拟合中线
            # self.fit_polynomial()

            if is_draw and find_left_corner and self.left_c is not None:
                cv2.circle(canvas, self.left_c, 4, (0, 255, 0), -1)
            if is_draw and find_right_corner and self.right_c is not None:
                cv2.circle(canvas, self.right_c, 4, (0, 255, 0), -1)

        except Exception as e:
            logging.error(f"Error occurred during getting side lines : {e}")

    def draw_line(self, canvas, fps=None, draw_fps=True):
        """绘制边线、中线

        Args:
            canvas: 用于绘制的画布图像
            fps: 实时FPS值
            draw_fps: 是否绘制实时FPS，默认为True

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

            # 绘制实时FPS（左上角）
            if draw_fps:
                fps_text = f"FPS: {fps:.2f}" if fps is not None else "FPS: 0.00"
                cv2.putText(
                    canvas,
                    fps_text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
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


def main_video():
    # 视频文件路径
    video_path = r"D:\programs\ucar_ws\src\vision_line\videos\test1.avi"

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error(f"Cannot open video: {video_path}")
        return

    logging.info(f"Video opened: {video_path}")

    imgprocess = ImageProcess()

    # 设置保存图片的目录（当前文件的上一级路径下的pictures文件夹）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(os.path.dirname(current_dir), "pictures")
    os.makedirs(save_dir, exist_ok=True)

    paused = False

    # FPS 计算：使用高精度计时器，逐帧更新 imgprocess.fps
    prev_t = None  # 上一帧时间戳（time.perf_counter）
    fps = 0.0
    dt = 0.0
    j = 0.0
    # 状态机参数
    straight_received = False
    right_received = True
    left_received = False
    corner_delay_s = 1.5  # 延时秒数，超过后启用拐点检测

    # 状态机变量
    state = ProcessState.IDLE
    t0 = None  # 指令开始时刻

    try:
        while True:
            frame = cap.read()[1]
            # img_sender.enqueue_image(frame)

            # 实时 FPS 计算
            now_t = time.perf_counter()
            if prev_t is not None:
                dt += now_t - prev_t
                j += 1
                if j % 10 == 0 and dt > 1e-6:
                    fps = 1.0 / dt * 10
                    logging.info(f"Current FPS: {fps:.2f}")
                    dt = 0.0
            prev_t = now_t

            # --- 状态机：状态转移 ---
            if state == ProcessState.IDLE:
                if straight_received:
                    state = ProcessState.STRAIGHT_TRACKING
                    t0 = time.perf_counter()
                    logging.info("State: IDLE -> STRAIGHT_TRACKING (straight received)")
                elif right_received:
                    state = ProcessState.RIGHT_TRACKING
                    logging.info("State: IDLE -> RIGHT_TRACKING (right received)")
                elif left_received:
                    state = ProcessState.LEFT_TRACKING
                    logging.info("State: IDLE -> LEFT_TRACKING (left received)")

            # --- 非 IDLE 状态才执行图像处理 ---
            else:
                # 保存当前帧到imgprocess
                imgprocess.frame = frame

                # 预处理
                binary_img = imgprocess.preprocess()

                if state not in (
                    ProcessState.RIGHT_TRACKING,
                    ProcessState.LEFT_TRACKING,
                ):
                    # STRAIGHT_TRACKING 超时检查
                    if state == ProcessState.STRAIGHT_TRACKING and t0 is not None:
                        if time.perf_counter() - t0 >= corner_delay_s:
                            state = ProcessState.CORNER
                            logging.info(
                                f"State: STRAIGHT_TRACKING -> CORNER (after {corner_delay_s}s)"
                            )

                    # 获取用于绘制的画布
                    canvas = imgprocess.return_frame()
                    # 根据状态决定 find_corner 参数
                    find_corner = state == ProcessState.CORNER

                    # 获取边线（传入canvas用于绘制调试信息）
                    imgprocess.get_side_line_task_2(
                        binary_img,
                        canvas,
                        is_draw=True,
                        find_corner=find_corner,
                    )

                    # CORNER 状态：处理完毕后检查是否进入 CROSS
                    if state == ProcessState.CORNER:
                        if imgprocess.judge_enter_cross_state(binary_img.shape, 0.75):
                            state = ProcessState.CROSS
                            logging.info(
                                "State: CORNER -> CROSS (dual corner detected)"
                            )

                    # CROSS 状态：检测停止线
                    if state == ProcessState.CROSS:
                        # 获取停止线
                        stop_mid = imgprocess.get_stop_line(
                            binary_img, is_draw=True, canvas=canvas
                        )

                        # 检查是否进入 TURNING
                        if stop_mid is not None and imgprocess.judge_enter_turning(
                            stop_mid, binary_img.shape
                        ):
                            state = ProcessState.TURNING
                            y_norm = stop_mid[1] / binary_img.shape[0]
                            logging.info(
                                f"State: CROSS -> TURNING (stop line at y={stop_mid[1]}, y_norm={y_norm:.2f})"
                            )

                    # 多项式拟合
                    imgprocess.fit_polynomial()

                    canvas = imgprocess.draw_line(canvas, fps, draw_fps=True)

                    # 显示二值化图像
                    cv2.imshow("binary", binary_img)
                    # img_sender.enqueue_image(binary_img, img_id=0)

                    # 显示处理结果
                    cv2.imshow("processed_img", canvas)
                    # 发送处理结果
                    # img_sender.enqueue_image(canvas, img_id=1)

                elif state in (ProcessState.RIGHT_TRACKING, ProcessState.LEFT_TRACKING):
                    canvas = imgprocess.return_frame()
                    # 如果是左转，就翻转一次得到正常视角，右转则不翻转，保持原视角
                    if state == ProcessState.LEFT_TRACKING:
                        binary_img = cv2.flip(binary_img, 1)
                        canvas = cv2.flip(canvas, 1)

                    imgprocess.get_side_line_task_1(
                        binary_img,
                        canvas,
                        is_draw=True,
                    )
                    # 多项式拟合
                    imgprocess.fit_polynomial()

                    canvas = imgprocess.draw_line(canvas, fps, draw_fps=True)

                    # 可视化
                    cv2.imshow("binary", binary_img)
                    cv2.imshow("processed_img", canvas)

                    # 发送图像
                    # img_sender.enqueue_image(binary_img, img_id=0)
                    # img_sender.enqueue_image(canvas, img_id=1)

            # 按键控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # q键退出
                logging.info("User quit")
                break
            elif key == ord(" "):  # 空格键暂停/继续
                paused = not paused
                if paused:
                    logging.info("Video paused")
                else:
                    # 恢复播放时重置 prev_t，避免首帧 FPS 因暂停时间被拉低
                    prev_t = None
                    logging.info("Video resumed")
            elif key == ord("s"):  # s键保存图片
                timestamp = int(time.perf_counter() * 1000)
                save_path = os.path.join(save_dir, f"captured_{timestamp}.jpg")
                cv2.imwrite(save_path, frame)
                logging.info(f"Captured image saved: {save_path}")

    except KeyboardInterrupt:
        logging.info("Interrupted by user , start exit...")

    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()


def main():

    # 设置保存图片的目录（当前文件的上一级路径下的pictures文件夹）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(os.path.dirname(current_dir), "pictures")
    os.makedirs(save_dir, exist_ok=True)

    imgprocess = ImageProcess()

    ip = "127.0.0.1"
    port = 12345
    img_sender = ImageSender(ip, port)

    # FPS 计算：使用高精度计时器，逐帧更新 imgprocess.fps
    prev_t = None  # 上一帧时间戳（time.perf_counter）
    fps = 0.0
    dt = 0.0
    j = 0.0
    # 状态机参数
    straight_received = False
    right_received = True
    left_received = False
    corner_delay_s = 1.5  # 延时秒数，超过后启用拐点检测

    # 状态机变量
    state = ProcessState.IDLE
    t0 = None  # 指令开始时刻

    try:
        # 连接服务器，开启发送线程
        img_sender.connect()

        img_sender.start_sending(2)

        cap = CameraCapture(0)  # 0表示默认摄像头
        if not cap.is_opened():
            logging.error("Cannot open camera.")
            return

        while True:
            frame = cap.get_picture()
            # img_sender.enqueue_image(frame)

            # 实时 FPS 计算
            now_t = time.perf_counter()
            if prev_t is not None:
                dt += now_t - prev_t
                j += 1
                if j % 10 == 0 and dt > 1e-6:
                    fps = 1.0 / dt * 10
                    logging.info(f"Current FPS: {fps:.2f}")
                    dt = 0.0
            prev_t = now_t

            # --- 状态机：状态转移 ---
            if state == ProcessState.IDLE:
                if straight_received:
                    state = ProcessState.STRAIGHT_TRACKING
                    t0 = time.perf_counter()
                    logging.info("State: IDLE -> STRAIGHT_TRACKING (straight received)")
                elif right_received:
                    state = ProcessState.RIGHT_TRACKING
                    logging.info("State: IDLE -> RIGHT_TRACKING (right received)")
                elif left_received:
                    state = ProcessState.LEFT_TRACKING
                    logging.info("State: IDLE -> LEFT_TRACKING (left received)")

            # --- 非 IDLE 状态才执行图像处理 ---
            else:
                # 保存当前帧到imgprocess
                imgprocess.frame = frame

                # 预处理
                binary_img = imgprocess.preprocess()
                if state not in (
                    ProcessState.RIGHT_TRACKING,
                    ProcessState.LEFT_TRACKING,
                ):
                    # STRAIGHT_TRACKING 超时检查
                    if state == ProcessState.STRAIGHT_TRACKING and t0 is not None:
                        if time.perf_counter() - t0 >= corner_delay_s:
                            state = ProcessState.CORNER
                            logging.info(
                                f"State: STRAIGHT_TRACKING -> CORNER (after {corner_delay_s}s)"
                            )

                    # 获取用于绘制的画布
                    canvas = imgprocess.return_frame()
                    # 根据状态决定 find_corner 参数
                    find_corner = state == ProcessState.CORNER

                    # 获取边线（传入canvas用于绘制调试信息）
                    imgprocess.get_side_line_task_2(
                        binary_img,
                        canvas,
                        is_draw=True,
                        find_corner=find_corner,
                    )

                    # CORNER 状态：处理完毕后检查是否进入 CROSS
                    if state == ProcessState.CORNER:
                        if imgprocess.judge_enter_cross_state(binary_img.shape, 0.75):
                            state = ProcessState.CROSS
                            logging.info(
                                "State: CORNER -> CROSS (dual corner detected)"
                            )

                    # CROSS 状态：检测停止线
                    if state == ProcessState.CROSS:
                        # 获取停止线
                        stop_mid = imgprocess.get_stop_line(
                            binary_img, is_draw=True, canvas=canvas
                        )

                        # 检查是否进入 TURNING
                        if stop_mid is not None and imgprocess.judge_enter_turning(
                            stop_mid, binary_img.shape
                        ):
                            state = ProcessState.TURNING
                            y_norm = stop_mid[1] / binary_img.shape[0]
                            logging.info(
                                f"State: CROSS -> TURNING (stop line at y={stop_mid[1]}, y_norm={y_norm:.2f})"
                            )

                    # 多项式拟合
                    imgprocess.fit_polynomial()

                    canvas = imgprocess.draw_line(canvas, fps, draw_fps=True)

                    # 显示二值化图像
                    # cv2.imshow("binary", binary_img)
                    img_sender.enqueue_image(binary_img, img_id=0)

                    # 显示处理结果
                    # cv2.imshow("processed_img", canvas)
                    # 发送处理结果
                    img_sender.enqueue_image(canvas, img_id=1)

                elif state in (ProcessState.RIGHT_TRACKING, ProcessState.LEFT_TRACKING):
                    canvas = imgprocess.return_frame()
                    # 如果是左转，就翻转一次得到正常视角，右转则不翻转，保持原视角
                    # if state == ProcessState.LEFT_TRACKING:
                    #     binary_img = cv2.flip(binary_img, 1)
                    #     canvas = cv2.flip(canvas, 1)

                    imgprocess.get_side_line_task_1(
                        binary_img,
                        canvas,
                        is_draw=True,
                    )
                    # 多项式拟合
                    imgprocess.fit_polynomial()

                    canvas = imgprocess.draw_line(canvas, fps, draw_fps=True)

                    # 可视化
                    cv2.imshow("binary", binary_img)
                    cv2.imshow("processed_img", canvas)

                    # 发送图像
                    # img_sender.enqueue_image(binary_img, img_id=0)
                    # img_sender.enqueue_image(canvas, img_id=1)

            # 按键控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # q键退出
                logging.info("User quit")
                break
            elif key == ord("s"):  # 保存图像
                timestamp = int(time.perf_counter() * 1000)
                save_path = os.path.join(save_dir, f"captured_{timestamp}.png")
                cv2.imwrite(save_path, frame)
                logging.info(f"Captured image saved: {save_path}")

    except KeyboardInterrupt:
        logging.info("Interrupted by user , start exit...")
    except Exception as e:
        logging.error(f"Error occurred during image process: {e}")

    finally:
        # 释放资源
        cap.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
