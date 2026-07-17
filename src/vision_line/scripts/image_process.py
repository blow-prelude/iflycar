import logging
import os
import time
from dataclasses import dataclass, field
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
    STRAIGHT_TRACKING = 1  # 直行循迹（单边或双边）
    RIGHT_TRACKING = 2  # 右转循迹
    LEFT_TRACKING = 3  # 左转循迹


class MissLineState(Enum):
    """转弯期间边线丢线状态机，对应 C++ 中的 MissLineState 枚举"""

    NO_MISS = 0  # 未丢线
    MISS = 1  # 丢线
    RECOVERED = 2  # 丢线后恢复，用于判断转弯结束


class MidLineMode(Enum):
    """mid_line 计算模式，对应 C++ 中的 MidLineMode 枚举"""

    MID_AVG = 0  # (supple_left + supple_right) / 2
    LEFT_OFFSET = 1  # supple_left + offset
    RIGHT_OFFSET = 2  # supple_right - offset


class SearchSide(Enum):
    """边线搜索范围，对应 C++ 中的 SearchSide 枚举"""

    BOTH = 0
    LEFT_ONLY = 1
    RIGHT_ONLY = 2


@dataclass
class ImageProcessConfig:
    """集中管理 ImageProcess 的所有可调参数"""

    # ---- 预处理 ----
    preprocess_max_h: int = 240
    preprocess_max_w: int = 320

    # ---- 边线搜索 ----
    x_continual: int = 10
    y_continual: int = 5
    search_offset: int = 30  # 起始搜索偏移量
    init_stable_count: int = 8
    miss_threshold: int = 3
    up_ratio: float = 0.55
    down_ratio: float = 0.90
    search_range_wide: int = 50  # 动态搜索窗口最大宽度
    search_range_narrow: int = 30  # 动态搜索窗口最小宽度
    search_range_threshold: float = 0.60  # 搜索窗口宽度调整阈值

    # ---- 拐点检测 ----
    corner_angle_high: int = 135
    corner_angle_low: int = 45
    corner_y_ratio: float = 0.70

    # ---- 停止线检测 ----
    stop_roi_y0: float = 0.70
    stop_roi_y1: float = 0.85
    stop_roi_x0: float = 0.30
    stop_roi_x1: float = 0.70
    stop_kernel_w: int = 15
    stop_min_width: int = 80

    # ---- 转弯判断 ----
    turning_enter_y_thresh: float = 0.78
    turning_end_x_diff: int = 20  # 转弯结束时两边线末端 x 坐标差异阈值
    turning_end_y_diff: int = 30  # 转弯结束时两边线末端 y 坐标差异阈值

    # ---- 多项式拟合（分段线性） ----
    fit_angle_thresh: float = 15.0
    fit_max_offset: int = 40  # 近端最大横向偏移量

    # ---- 插值 & 填充 ----
    interp_dx_thresh: int = 6
    interp_dy_thresh: int = 3
    fill_down_ratio: float = 0.90

    # ---- 透视变换矩阵 ----
    perspective_matrix: list = field(
        default_factory=lambda: [
            [-0.498345, -1.637252, 251.246087],
            [-0.021773, 0.349689, -81.638093],
            [-0.000241, -0.009225, 1.000000],
        ]
    )

    # ---- 拟合曲线上的目标点索引（负数表示从末尾倒数，与 image_process_ros 对齐） ----
    straight_target_p_index = -10
    tracking2_target_p_index = -48
    left_target_p_index = -25

    # ---- TURNING/单边循迹时用单边线生成 mid_line 的横向偏移（像素） ----
    turning_mid_offset: int = 40


def compute_target_x_error(line_points, processed_shape, original_shape, index):
    """计算拟合中线上指定索引点的 x_error 与 y_pixel（缩放到原始分辨率）。

    与 image_process_ros.build_vision_line_msg_by_index 等价，但不构造 ROS 消息，
    供离线 main/mian_video 用于 TURNING -> TRACKING2 的阈值判断。

    Returns:
        (x_error, y_pixel): 输入无效时返回 (0.0, -1.0)
    """
    if (
        line_points is None
        or len(line_points) == 0
        or processed_shape is None
        or original_shape is None
        or len(processed_shape) < 2
        or len(original_shape) < 2
        or len(line_points) < abs(index) + 1
    ):
        return 0.0, -1.0

    proc_h, proc_w = processed_shape[:2]
    orig_h, orig_w = original_shape[:2]

    if proc_h <= 0 or proc_w <= 0 or orig_h <= 0 or orig_w <= 0:
        return 0.0, -1.0

    scale_x = float(orig_w) / float(proc_w)
    scale_y = float(orig_h) / float(proc_h)

    best_x, best_y = int(line_points[index, 0]), int(line_points[index, 1])
    x_raw = best_x * scale_x
    y_raw = best_y * scale_y
    x_error = x_raw - (orig_w / 2.0)
    return float(x_error), float(y_raw)


class ImageProcess:
    def __init__(self, img_path=None, img=None, config=None):
        """
        初始化图像处理对象

        Args:
            img_path: 图片路径，如果提供则从图片读取
            img: 图片数据
            config: ImageProcessConfig 实例，为 None 时使用默认参数
        """

        self.cfg = config or ImageProcessConfig()

        self.img_path = img_path
        self.frame = img
        self.canvas = None
        _empty = np.empty((0, 2), dtype=np.int32)
        self.left_line = _empty.copy()
        self.right_line = _empty.copy()
        self.supple_left_line = _empty.copy()
        self.supple_right_line = _empty.copy()
        self.mid_line = _empty.copy()
        self.fit_mid_line = _empty.copy()

        self.left_c = None  # 本帧左边线拐点 (x, y)，未检测到时为 None
        self.right_c = None  # 本帧右边线拐点 (x, y)，未检测到时为 None

        # 上一帧的边线信息（用于指导当前帧搜索）
        self.prev_left_line = _empty.copy()
        self.prev_right_line = _empty.copy()
        self.prev_supple_left_line = _empty.copy()
        self.prev_supple_right_line = _empty.copy()

        self.perspective_matrix = np.array(
            self.cfg.perspective_matrix, dtype=np.float64
        )

        # mid_line 计算模式（单边循迹时使用 LEFT_OFFSET/RIGHT_OFFSET）
        self.mid_line_mode_ = MidLineMode.MID_AVG

    def set_mid_line_mode(self, mode: MidLineMode):
        """设置 mid_line 计算模式"""
        self.mid_line_mode_ = mode

    def get_mid_line_mode(self) -> MidLineMode:
        """获取 mid_line 计算模式"""
        return self.mid_line_mode_

    def preprocess(self, frame):
        try:
            if self.img_path is not None:
                # 从图片文件读取
                frame = cv2.imread(self.img_path)

            if self.frame is not None:
                frame = self.frame

            if frame is not None:
                # 如果图片太大，按比例缩小
                if (
                    frame.shape[0] >= self.cfg.preprocess_max_h
                    or frame.shape[1] >= self.cfg.preprocess_max_w
                ):
                    h, w = frame.shape[:2]
                    scale = min(
                        self.cfg.preprocess_max_h / h, self.cfg.preprocess_max_w / w
                    )
                    new_h = int(h * scale)
                    new_w = int(w * scale)
                    frame = cv2.resize(
                        frame, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )
                    self.frame = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
        if matrix is None:
            matrix = self.perspective_matrix

        if matrix is None:
            raise ValueError(
                "[perspective_transform]:Perspective transform matrix is None"
            )

        if img is None:
            raise ValueError("[perspective_transform]:Input image is None")

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

        logging.debug(f"Perspective transform completed, output size: {output_size}")
        return transformed_img

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
            prev_line: 上一帧的边线点 numpy数组 (N, 2)
            img_width: 图像宽度
            is_left: 是否为左边线（True=左边线，False=右边线）

        Returns:
            int: 搜索起点的x坐标，如果没有上一帧信息则返回中线位置
        """
        if len(prev_line) == 0:
            return img_width // 2
        best_idx = np.argmin(np.abs(prev_line[:, 1] - y_coord))
        prev_x = int(prev_line[best_idx, 0])

        if is_left:
            start_x = prev_x + self.cfg.search_offset
        else:
            start_x = prev_x - self.cfg.search_offset
        return max(0, min(start_x, img_width - 1))

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
                if len(stable_buf) >= self.cfg.init_stable_count:
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
        """
        对边线点进行线性插值，填充点之间的空隙
        Args:  line_points: 边线点列表 [(x, y), ...]
        Returns: 插值后的边线点列表 [(x, y), ...]

        使用np向量化，加快效率
        """
        if len(line_points) < 2:
            return np.asarray(line_points, dtype=np.int32).reshape(-1, 2).copy()

        pts = np.array(line_points, dtype=np.int32)

        p1 = pts[:-1]
        p2 = pts[1:]

        dx = np.abs(p2[:, 0] - p1[:, 0])
        dy = np.abs(p2[:, 1] - p1[:, 1])

        need_interp = (dx > self.cfg.interp_dx_thresh) | (
            dy > self.cfg.interp_dy_thresh
        )

        result = []

        for i in range(len(p1)):
            x1, y1 = p1[i]
            x2, y2 = p2[i]

            result.append((x1, y1))

            if not need_interp[i]:
                continue

            num_points = max(dx[i] // 3, dy[i] // 2, 2)

            t = np.linspace(0, 1, num_points + 2)[1:-1]

            xs = (x1 + (x2 - x1) * t).astype(np.int32)
            ys = (y1 + (y2 - y1) * t).astype(np.int32)

            # 向量化过滤
            mask = (ys > min(y1, y2)) & (ys < max(y1, y2))

            interp_pts = np.stack([xs[mask], ys[mask]], axis=1)

            if len(interp_pts) > 0:
                result.extend(map(tuple, interp_pts))

        result.append(tuple(pts[-1]))
        return np.array(result, dtype=np.int32).reshape(-1, 2)

    def _fill_boundary(self, left_line, right_line, img_shape):
        """将边线延伸到指定y位置，处理丢线情况

        三种情况：
        1. 两边都有线：插值后填充到底部
        2. 一边丢线：存在的一边插值填充，缺失的一边用图像边界替代

        Args:
            left_line: 左边线 numpy数组 (N, 2)
            right_line: 右边线 numpy数组 (N, 2)
            img_shape: 图像形状 (height, width)

        Returns:
            (填充后的左边线, 填充后的右边线) 均为 numpy数组
        """
        img_h, img_w = img_shape[0], img_shape[1]
        bottom_y_limit = int(self.cfg.fill_down_ratio * img_h)

        # 两边都有线：插值 + 填充到底部
        if len(left_line) > 0 and len(right_line) > 0:
            supple_left = self._linear_interpolation(left_line)
            supple_right = self._linear_interpolation(right_line)

            bottom_y = int(supple_left[0, 1])
            if bottom_y < bottom_y_limit:
                ys = np.arange(bottom_y_limit, bottom_y, -2, dtype=np.int32)
                xs = np.zeros_like(ys)
                supple_left = np.vstack([np.stack([xs, ys], axis=1), supple_left])

            bottom_y = int(supple_right[0, 1])
            if bottom_y < bottom_y_limit:
                ys = np.arange(bottom_y_limit, bottom_y, -2, dtype=np.int32)
                xs = np.full_like(ys, img_w - 1)
                supple_right = np.vstack([np.stack([xs, ys], axis=1), supple_right])

            return supple_left, supple_right

        # 左边丢线，右边存在
        if len(left_line) == 0 and len(right_line) > 0:
            supple_right = self._linear_interpolation(right_line)
            bottom_y = int(supple_right[0, 1])
            if bottom_y < bottom_y_limit:
                ys = np.arange(bottom_y_limit, bottom_y, -2, dtype=np.int32)
                xs = np.full_like(ys, img_w - 1)
                supple_right = np.vstack([np.stack([xs, ys], axis=1), supple_right])

            ys_all = supple_right[:, 1]
            boundary_left = np.stack([np.zeros_like(ys_all), ys_all], axis=1)
            return boundary_left, supple_right

        # 右边丢线，左边存在
        if len(right_line) == 0 and len(left_line) > 0:
            supple_left = self._linear_interpolation(left_line)
            bottom_y = int(supple_left[0, 1])
            if bottom_y < bottom_y_limit:
                ys = np.arange(bottom_y_limit, bottom_y, -2, dtype=np.int32)
                xs = np.zeros_like(ys)
                supple_left = np.vstack([np.stack([xs, ys], axis=1), supple_left])

            ys_all = supple_left[:, 1]
            boundary_right = np.stack([np.full_like(ys_all, img_w - 1), ys_all], axis=1)
            return supple_left, boundary_right

        return left_line, right_line

    def _fill_missing_line(self, img_shape):
        """当左线或右线缺失时，用图像边界替代缺失边线

        已存在的那条线会经过插值和底部边界填充，确保与边界线 y 对齐。
        边界线使用已存在线的完整 y 列表（而非固定步长），保证一一对应。

        Args:
            img_shape: 图像形状 (height, width, ...)

        Returns:
            bool: 是否填充了缺失线
        """
        img_h, img_w = img_shape[0], img_shape[1]

        # 左线缺失，右线存在
        if len(self.left_line) == 0 and len(self.right_line) > 0:
            supple_right = self._linear_interpolation(self.right_line)
            bottom_y = int(supple_right[0, 1])
            if bottom_y < img_h - 1:
                ys = np.arange(
                    img_h - 1, bottom_y, -2, dtype=np.int32
                )  # 使用np生成y坐标列表
                xs = np.full_like(ys, img_w - 1)
                bottom_pts = np.stack([xs, ys], axis=1)
                supple_right = np.vstack([bottom_pts, supple_right])

            ys_all = supple_right[:, 1]
            boundary_left = np.stack([np.zeros_like(ys_all), ys_all], axis=1)

            self.supple_left_line = boundary_left
            self.supple_right_line = supple_right
            return True

        if len(self.right_line) == 0 and len(self.left_line) > 0:
            supple_left = self._linear_interpolation(self.left_line)
            bottom_y = int(supple_left[0, 1])

            if bottom_y < img_h - 1:
                ys = np.arange(img_h - 1, bottom_y, -2, dtype=np.int32)
                xs = np.zeros_like(ys)
                bottom_pts = np.stack([xs, ys], axis=1)
                supple_left = np.vstack([bottom_pts, supple_left])

            ys_all = supple_left[:, 1]
            boundary_right = np.stack([np.full_like(ys_all, img_w - 1), ys_all], axis=1)

            self.supple_right_line = boundary_right
            self.supple_left_line = supple_left
            return True

        return False

    def fit_polynomial(self):
        """分段线性拟合中线：近端和远端各用一次函数拟合，
        当两段斜率差异较大时，根据远端方向给近端施加横向偏移。
        """
        self.fit_mid_line = np.empty((0, 2), dtype=np.int32)
        if len(self.mid_line) < 4:
            logging.warning("Not enough points for polynomial fitting")
            return

        try:
            y_points = self.mid_line[:, 1].astype(np.float64)
            x_points = self.mid_line[:, 0].astype(np.float64)

            # 按 y 中值分为远端（小 y，图像上方）和近端（大 y，图像下方）
            y_mid = (y_points.min() + y_points.max()) / 2.0
            near_mask = y_points >= y_mid
            far_mask = ~near_mask

            if np.sum(near_mask) < 2 or np.sum(far_mask) < 2:
                # 某段点数不足，退化为整体一次拟合
                coeff = np.polyfit(y_points, x_points, 1)
                ys = np.arange(
                    int(y_points.min()), int(y_points.max()) + 1, dtype=np.float64
                )
                xs = np.polyval(coeff, ys).astype(np.int32)
                self.fit_mid_line = np.stack([xs, ys.astype(np.int32)], axis=1)
                return

            # 分段一次拟合 x = k*y + b
            near_y, near_x = y_points[near_mask], x_points[near_mask]
            far_y, far_x = y_points[far_mask], x_points[far_mask]

            near_coeff = np.polyfit(near_y, near_x, 1)  # [k_near, b_near]
            far_coeff = np.polyfit(far_y, far_x, 1)  # [k_far, b_far]

            k_near, k_far = near_coeff[0], far_coeff[0]

            # 计算两直线夹角
            angle = self.get_angle_k(k_near, k_far)

            offset = 0

            if angle > self.cfg.fit_angle_thresh:
                offset = int(
                    -np.sign(k_far) * min(angle / 45.0, 1.0) * self.cfg.fit_max_offset
                )

            # 生成近端拟合点（含偏移）
            near_ys = np.arange(
                int(near_y.min()), int(near_y.max()) + 1, dtype=np.float64
            )
            near_xs = np.polyval(near_coeff, near_ys).astype(np.int32) + offset

            # 生成远端拟合点
            far_ys = np.arange(int(far_y.min()), int(far_y.max()) + 1, dtype=np.float64)
            far_xs = np.polyval(far_coeff, far_ys).astype(np.int32)

            near_pts = np.stack([near_xs, near_ys.astype(np.int32)], axis=1)
            far_pts = np.stack([far_xs, far_ys.astype(np.int32)], axis=1)

            self.fit_mid_line = np.vstack([far_pts, near_pts])

            logging.debug(
                f"Two-segment fit: near_k={k_near:.3f}, far_k={k_far:.3f}, "
                f"angle={angle:.1f}°, offset={offset}px"
            )
        except Exception as e:
            logging.error(f"Error occurred during polynomial fitting: {e}")

    def fit_polynomial2(self):
        """根据self.mid_line的原始值，用二次函数拟合曲线，返回曲线上的点的numpy数组 self.fit_mid_line"""
        self.fit_mid_line = np.empty((0, 2), dtype=np.int32)
        if len(self.mid_line) < 3:
            logging.warning("Not enough points for polynomial fitting")
            return

        try:
            y_points = self.mid_line[:, 1].astype(np.float64)
            x_points = self.mid_line[:, 0].astype(np.float64)

            coefficients = np.polyfit(y_points, x_points, 2)

            y_min = int(y_points.min())
            y_max = int(y_points.max())
            ys = np.arange(y_min, y_max + 1, dtype=np.float64)
            xs = np.polyval(coefficients, ys).astype(np.int32)
            self.fit_mid_line = np.stack([xs, ys.astype(np.int32)], axis=1)

            logging.debug(
                f"Polynomial fitting completed: {len(self.fit_mid_line)} points, "
                f"coefficients: {coefficients}"
            )
        except Exception as e:
            logging.error(f"Error occurred during polynomial fitting: {e}")

    def judge_enter_cross_state(self, img_shape):
        """判断是否满足进入 CROSS 状态的条件

        Args:
            img_shape: 图像形状 (height, width)

        Returns:
            bool: 满足条件返回 True，否则 False
        """
        if self.left_c is None or self.right_c is None:
            return False
        img_h = img_shape[0]
        return (
            self.left_c[1] >= img_h * self.cfg.corner_y_ratio
            or self.right_c[1] >= img_h * self.cfg.corner_y_ratio
        )

    def judge_enter_turning(self, stop_mid, img_shape):
        """判断是否应该进入 TURNING 状态

        Args:
            stop_mid: 停止线中点坐标 (x, y) 或 None
            img_shape: 图像形状 (h, w, ...)

        Returns:
            bool: True 表示应该进入 TURNING 状态
        """
        if stop_mid is None:
            return False

        y_norm = stop_mid[1] / img_shape[0]
        return y_norm > self.cfg.turning_enter_y_thresh

    def judge_turning_end(self, img_shape, miss_line=[MissLineState.NO_MISS]):
        """判断转弯是否结束：一开始两边都不丢线，然后一边丢线一边不丢线，最后两边都不丢线

        Args:
            img_shape: 图像形状 (h, w, ...)
            miss_line: 转弯期间丢线状态 [MissLineState]

        Returns:
            bool: True 表示转弯结束，可以进入 TRACKING2 状态
        """
        if len(self.left_line) == 0 or len(self.right_line) == 0:
            miss_line[0] = MissLineState.MISS
            return False

        x_diff = abs(int(self.right_line[-1, 0]) - int(self.left_line[-1, 0]))
        if miss_line[0] == MissLineState.MISS and x_diff <= self.cfg.turning_end_x_diff:
            miss_line[0] = MissLineState.RECOVERED
            return True

        if x_diff <= self.cfg.turning_end_x_diff:
            miss_line[0] = MissLineState.MISS
            return False

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
        roi_y0 = int(h * self.cfg.stop_roi_y0)
        roi_y1 = int(h * self.cfg.stop_roi_y1)
        roi_x0 = int(w * self.cfg.stop_roi_x0)
        roi_x1 = int(w * self.cfg.stop_roi_x1)
        roi = binary_img[roi_y0 : roi_y1 + 1, roi_x0 : roi_x1 + 1]

        logging.debug(f"ROI: y=[{roi_y0}, {roi_y1}], x=[{roi_x0}, {roi_x1}]")

        # 横向形态学削弱斜线
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.cfg.stop_kernel_w, 1))
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)

        contours = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        if len(contours) == 0:
            logging.debug("No contours found in stop line ROI")
            return None

        # 找最像“横线”的轮廓
        best = max(contours, key=lambda c: cv2.boundingRect(c)[2])  # 按宽度选

        # M = cv2.moments(best)
        # if M["m00"] != 0:
        #     mid_x = int(M["m10"] / M["m00"])
        #     mid_y = int(M["m01"] / M["m00"])

        x, y, w, h = cv2.boundingRect(best)

        if w <= self.cfg.stop_min_width:
            return None
        # 还原到原图像
        x += roi_x0
        y += roi_y0

        mid_x = x + w // 2
        mid_y = y + h // 2

        logging.debug(f"find stop line ,mid:({mid_x}, {mid_y})")

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

    def offset_line(self, line, offset):
        """将边线整体在 x 方向偏移指定像素数

        Args:
            line: 边线点 numpy 数组 (N, 2)，每行为 (x, y)
            offset: x 方向偏移量（像素），正数向右，负数向左

        Returns:
            numpy 数组: 偏移后的边线，形状与输入相同
        """
        if len(line) == 0:
            return line.copy()
        result = line.copy()
        result[:, 0] += offset
        return result

    def get_side_line_task_1(self, img, canvas, is_draw=False):
        """从图像的中线往两边搜索，获取赛道边线

        使用同帧逐行递推决定搜索起点，搜索窗口越往上越窄。

        Args:
            img: 输入的二值化图像
            canvas: 用于绘制的画布图像
            is_draw: 是否在canvas上绘制调试信息，默认为False
        """
        mid_x = img.shape[1] // 2
        img_h, img_w = img.shape[:2]
        try:
            local_left_line = []
            local_right_line = []
            _empty = np.empty((0, 2), dtype=np.int32)

            # 逐行递推的搜索起点（初始为 mid_x）
            prev_row_left_x = mid_x
            prev_row_right_x = mid_x

            # miss 计数（只在 stable 后计数）
            left_miss_count = 0
            right_miss_count = 0

            # 稳定点缓冲区及标志
            left_stable_buf = []
            right_stable_buf = []
            left_stable = [False]
            right_stable = [False]

            diff = np.diff(img == 0, axis=1)  # 计算行内黑白跳变  右-左

            for y in range(
                int(img_h * self.cfg.down_ratio), int(img_h * self.cfg.up_ratio), -1
            ):
                row_diff = diff[y]

                # 动态搜索窗口：二段阶梯
                y_norm = y / img_h
                cur_range = (
                    self.cfg.search_range_wide
                    if y_norm > self.cfg.search_range_threshold
                    else self.cfg.search_range_narrow
                )

                # --- 左侧赛道线 ---
                if left_stable[0]:
                    search_start_left = min(prev_row_left_x + cur_range, img_w - 1)
                    search_end_left = max(0, prev_row_left_x - cur_range)
                else:
                    search_start_left = mid_x - self.cfg.search_offset
                    search_end_left = 0

                if is_draw:
                    cv2.circle(canvas, (search_start_left, y), 1, (0, 255, 255), -1)
                    cv2.circle(canvas, (search_end_left, y), 1, (0, 255, 255), -1)

                candidates = np.where(row_diff[search_end_left:search_start_left] == 1)[
                    0
                ]
                left_added = False
                if len(candidates) > 0:
                    x = search_end_left + candidates[-1]
                    left_added = self._add_point_with_stable_start(
                        local_left_line,
                        (x, y),
                        left_stable_buf,
                        left_stable,
                        self.cfg.x_continual,
                        self.cfg.y_continual,
                    )
                    # 如果当前行的点没有被加入正式边线，则下一行的搜索起点不更新；反之才更新
                    if left_added:
                        prev_row_left_x = x

                # miss 计数（只在 stable 后）
                if left_stable[0] and not left_added:
                    left_miss_count += 1
                else:
                    left_miss_count = 0

                if left_miss_count >= self.cfg.miss_threshold:
                    prev_row_left_x = mid_x - self.cfg.search_offset
                    left_miss_count = 0

                # --- 右侧赛道线 ---
                if right_stable[0]:
                    search_start_right = max(
                        0, min(prev_row_right_x - cur_range, img_w - 1)
                    )
                    search_end_right = min(img_w - 1, prev_row_right_x + cur_range)
                else:
                    search_start_right = mid_x + self.cfg.search_offset
                    search_end_right = img_w - 1

                if y % 10 == 0:
                    logging.info(
                        f"y={y}, search range: left [{search_end_left}, {search_start_left}], "
                        f"right [{search_start_right}, {search_end_right}]"
                    )

                if is_draw:
                    cv2.circle(canvas, (search_start_right, y), 1, (255, 255, 0), -1)
                    cv2.circle(canvas, (search_end_right, y), 1, (255, 255, 0), -1)

                candidates = np.where(
                    row_diff[search_start_right:search_end_right] == 1
                )[0]
                right_added = False
                if len(candidates) > 0:
                    x = search_start_right + candidates[0]
                    right_added = self._add_point_with_stable_start(
                        local_right_line,
                        (x, y),
                        right_stable_buf,
                        right_stable,
                        self.cfg.x_continual,
                        self.cfg.y_continual,
                    )
                    if right_added:
                        prev_row_right_x = x

                # miss 计数（只在 stable 后）
                if right_stable[0] and not right_added:
                    right_miss_count += 1
                else:
                    right_miss_count = 0

                if right_miss_count >= self.cfg.miss_threshold:
                    prev_row_right_x = mid_x + self.cfg.search_offset
                    right_miss_count = 0

            # 转换为 numpy 数组
            self.left_line = (
                np.array(local_left_line, dtype=np.int32).reshape(-1, 2)
                if local_left_line
                else _empty.copy()
            )
            self.right_line = (
                np.array(local_right_line, dtype=np.int32).reshape(-1, 2)
                if local_right_line
                else _empty.copy()
            )
            self.supple_left_line = _empty.copy()
            self.supple_right_line = _empty.copy()
            self.fit_mid_line = _empty.copy()

            # 插值 + 填充边线（同时处理丢线情况）
            self.supple_left_line, self.supple_right_line = self._fill_boundary(
                self.left_line, self.right_line, img.shape
            )

            # 使用优化后的边线计算中线（向量化）
            n = min(len(self.supple_left_line), len(self.supple_right_line))
            if n > 0:
                mid_xs = (
                    self.supple_left_line[:n, 0] + self.supple_right_line[:n, 0]
                ) // 2
                mid_ys = self.supple_left_line[:n, 1]
                self.mid_line = np.stack([mid_xs, mid_ys], axis=1)
            else:
                self.mid_line = _empty.copy()

        except Exception as e:
            logging.error(f"Error occurred during getting side lines : {e}")
        finally:
            # 更新上一帧的边线信息
            self._update_prev_frame_lines()

    def get_side_line_task_2(
        self,
        img,
        canvas,
        is_draw=False,
        find_corner=False,
        side=SearchSide.BOTH,
    ):
        """从图像的中线往两边搜索，获取赛道边线

        Args:
            img: 输入的二值化图像
            canvas: 用于绘制的画布图像
            is_draw: 是否在canvas上绘制调试信息，默认为False
            find_corner: 是否搜寻拐点
            side: 边线搜索范围（BOTH/LEFT_ONLY/RIGHT_ONLY）
        """
        # 从图像中间向两边搜索，获取边线

        # 拐点检测
        left_nxt_p, left_cur_p, left_pre_p = None, None, None
        right_nxt_p, right_cur_p, right_pre_p = None, None, None
        find_left_corner, find_right_corner = False, False

        mid_x = int(img.shape[1] // 2)
        img_h, img_w = img.shape[:2]

        try:
            local_left_line = []
            local_right_line = []
            _empty = np.empty((0, 2), dtype=np.int32)
            self.left_c = None
            self.right_c = None

            # 逐行递推的搜索起点（初始为 mid_x）
            prev_row_left_x = mid_x
            prev_row_right_x = mid_x

            # miss 计数（只在 stable 后计数）
            left_miss_count = 0
            right_miss_count = 0

            # 稳定点缓冲区及标志
            left_stable_buf = []
            right_stable_buf = []
            left_stable = [False]
            right_stable = [False]

            diff = np.diff(img == 0, axis=1)  # 计算行内黑白跳变  右-左
            # cv2.imshow("diff", (diff != 0).astype(np.uint8) * 255)  # 显示发生跳变的地方

            # 从图像下方（靠近车辆）开始搜索
            for y in range(
                int(img.shape[0] * self.cfg.down_ratio),
                int(img.shape[0] * self.cfg.up_ratio),
                -1,
            ):
                row_diff = diff[y]

                # 动态搜索窗口：二段阶梯
                y_norm = y / img_h
                cur_range = (
                    self.cfg.search_range_wide
                    if y_norm > self.cfg.search_range_threshold
                    else self.cfg.search_range_narrow
                )

                # --- 左侧赛道线 ---
                left_added = False
                if side != SearchSide.RIGHT_ONLY:
                    if left_stable[0]:
                        search_start_left = min(prev_row_left_x + cur_range, img_w - 1)
                        search_end_left = max(0, prev_row_left_x - cur_range)
                    # 默认从中间偏左一直搜索到左边界
                    else:
                        search_start_left = mid_x - self.cfg.search_offset
                        search_end_left = 0

                    if is_draw:
                        cv2.circle(canvas, (search_start_left, y), 1, (0, 255, 255), -1)
                        cv2.circle(canvas, (search_end_left, y), 1, (0, 255, 255), -1)

                    candidates = np.where(row_diff[search_end_left:search_start_left] == 1)[
                        0
                    ]

                    if len(candidates) > 0:
                        x = candidates[-1] + search_end_left

                        # 先进行稳定点检测
                        left_added = self._add_point_with_stable_start(
                            local_left_line,
                            (x, y),
                            left_stable_buf,
                            left_stable,
                            self.cfg.x_continual,
                            self.cfg.y_continual,
                        )

                        # 如果当前行的点没有被加入正式边线，则下一行的搜索起点不更新；反之才更新
                        if left_added:
                            prev_row_left_x = x
                            # 只有稳定点才参与拐点检测
                            if find_corner and not find_left_corner:
                                left_nxt_p = (x, y)
                                if left_cur_p is not None and left_pre_p is not None:
                                    angle = self.get_angle_p(
                                        left_nxt_p, left_cur_p, left_pre_p
                                    )
                                    logging.debug(
                                        f"left line angle: {angle} , pre_p: {left_pre_p},  cur_p: {left_cur_p} , nxt_p: {left_nxt_p}"
                                    )
                                    if (
                                        self.cfg.corner_angle_low
                                        < angle
                                        < self.cfg.corner_angle_high
                                    ):
                                        logging.debug(
                                            f"slope mutation , angle: {angle} ,pre_p:{left_pre_p} , cur_p: {left_cur_p} , nxt_p: {left_nxt_p} "
                                        )
                                        self.left_c = left_cur_p
                                        find_left_corner = True
                                # 更新点
                                left_pre_p = left_cur_p
                                left_cur_p = left_nxt_p

                        # miss 计数（只在 stable 后）,如果连续丢失多个点就恢复默认搜索范围
                        if left_stable[0] and not left_added:
                            left_miss_count += 1
                        else:
                            left_miss_count = 0

                        if left_miss_count >= self.cfg.miss_threshold:
                            prev_row_left_x = mid_x - self.cfg.search_offset
                            left_miss_count = 0

                # 右线
                right_added = False
                if side != SearchSide.LEFT_ONLY:
                    if right_stable[0]:
                        search_start_right = max(
                            0, min(prev_row_right_x - cur_range, img_w - 1)
                        )
                        search_end_right = min(img_w - 1, prev_row_right_x + cur_range)
                    else:
                        search_start_right = mid_x + self.cfg.search_offset
                        search_end_right = img_w - 1

                    if is_draw:
                        cv2.circle(canvas, (search_start_right, y), 1, (255, 255, 0), -1)
                        cv2.circle(canvas, (search_end_right, y), 1, (255, 255, 0), -1)

                    candidates = np.where(
                        row_diff[search_start_right:search_end_right] == 1
                    )[0]
                    if len(candidates) > 0:
                        x = candidates[0] + search_start_right

                        # 先进行稳定点检测
                        right_added = self._add_point_with_stable_start(
                            local_right_line,
                            (x, y),
                            right_stable_buf,
                            right_stable,
                            self.cfg.x_continual,
                            self.cfg.y_continual,
                        )
                        if right_added:
                            prev_row_right_x = x

                            # 只有稳定点才参与拐点检测
                            if find_corner and find_right_corner is False:
                                right_nxt_p = (x, y)
                                if right_cur_p is not None and right_pre_p is not None:
                                    angle = self.get_angle_p(
                                        right_nxt_p, right_cur_p, right_pre_p
                                    )
                                    logging.debug(
                                        f"right line angle: {angle} , pre_p: {right_pre_p},  cur_p: {right_cur_p} , nxt_p: {right_nxt_p}"
                                    )
                                    if (
                                        self.cfg.corner_angle_low
                                        < angle
                                        < self.cfg.corner_angle_high
                                    ):
                                        logging.debug(
                                            f"slope mutation , angle: {angle} ,pre_p:{right_pre_p} cur_p: {right_cur_p} , nxt_p: {right_nxt_p} "
                                        )
                                        self.right_c = right_cur_p
                                        find_right_corner = True
                            # 更新点
                            right_pre_p = right_cur_p
                            right_cur_p = right_nxt_p

                        # miss 计数（只在 stable 后）
                        if right_stable[0] and not right_added:
                            right_miss_count += 1
                        else:
                            right_miss_count = 0

                        if right_miss_count >= self.cfg.miss_threshold:
                            prev_row_right_x = mid_x + self.cfg.search_offset
                            right_miss_count = 0

            # 转换为 numpy 数组
            self.left_line = (
                np.array(local_left_line, dtype=np.int32).reshape(-1, 2)
                if local_left_line
                else _empty.copy()
            )
            self.right_line = (
                np.array(local_right_line, dtype=np.int32).reshape(-1, 2)
                if local_right_line
                else _empty.copy()
            )
            self.supple_left_line = _empty.copy()
            self.supple_right_line = _empty.copy()
            self.fit_mid_line = _empty.copy()

            # 插值 + 填充边线（同时处理丢线情况）
            self.supple_left_line, self.supple_right_line = self._fill_boundary(
                self.left_line, self.right_line, img.shape
            )

            # 用优化后的边线计算中线（根据 mid_line_mode_ 选择计算方式）
            if self.mid_line_mode_ == MidLineMode.LEFT_OFFSET:
                if len(self.supple_left_line) > 0:
                    mid_xs = self.supple_left_line[:, 0] + self.cfg.turning_mid_offset
                    mid_ys = self.supple_left_line[:, 1]
                    self.mid_line = np.stack([mid_xs, mid_ys], axis=1)
                else:
                    self.mid_line = _empty.copy()
            elif self.mid_line_mode_ == MidLineMode.RIGHT_OFFSET:
                if len(self.supple_right_line) > 0:
                    mid_xs = (
                        self.supple_right_line[:, 0] - self.cfg.turning_mid_offset
                    )
                    mid_ys = self.supple_right_line[:, 1]
                    self.mid_line = np.stack([mid_xs, mid_ys], axis=1)
                else:
                    self.mid_line = _empty.copy()
            else:  # MID_AVG
                n = min(len(self.supple_left_line), len(self.supple_right_line))
                if n > 0:
                    mid_xs = (
                        self.supple_left_line[:n, 0] + self.supple_right_line[:n, 0]
                    ) // 2
                    mid_ys = self.supple_left_line[:n, 1]
                    self.mid_line = np.stack([mid_xs, mid_ys], axis=1)
                else:
                    self.mid_line = _empty.copy()

            # 多项式拟合中线
            # self.fit_polynomial()

            if is_draw and find_left_corner and self.left_c is not None:
                cv2.circle(canvas, self.left_c, 4, (0, 255, 0), -1)
            if is_draw and find_right_corner and self.right_c is not None:
                cv2.circle(canvas, self.right_c, 4, (0, 255, 0), -1)

        except Exception as e:
            logging.error(f"Error occurred during getting side lines : {e}")

    def draw_line(self, canvas, fps=float("inf"), state=None):
        """绘制边线、中线

        Args:
            canvas: 用于绘制的画布图像
            fps: 实时FPS值
            state: 当前状态 (ProcessState)

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
            if fps != float("inf"):
                fps_text = f"FPS: {fps:.2f}"
                cv2.putText(
                    canvas,
                    fps_text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 128),
                    2,
                    cv2.LINE_AA,
                )

            # 绘制当前状态（右上角）
            if state is not None:
                state_text = state.name
                (tw, th), _ = cv2.getTextSize(
                    state_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.putText(
                    canvas,
                    state_text,
                    (canvas.shape[1] - tw - 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            logging.debug(
                f"length of left_line: {len(self.supple_left_line)} , lenth of right_line: {len(self.supple_right_line)}"
            )

            # 绘制中线
            cv2.line(
                canvas,
                (canvas.shape[1] // 2 - 1, 0),
                (canvas.shape[1] // 2 - 1, canvas.shape[0] - 1),
                (255, 0, 127),
                1,
            )

            # 绘制优化后的边线和中线
            for pt in self.supple_left_line.tolist():
                cv2.circle(canvas, pt, 2, (0, 0, 255), -1)
            for pt in self.supple_right_line.tolist():
                cv2.circle(canvas, pt, 2, (0, 0, 255), -1)
            for pt in self.mid_line.tolist():
                cv2.circle(canvas, pt, 2, (255, 0, 0), -1)
            for pt in self.fit_mid_line.tolist():
                cv2.circle(canvas, pt, 2, (255, 255, 255), -1)
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
    straight_received = True
    right_received = False
    left_received = False

    # STRAIGHT_TRACKING 走哪一边：LEFT_ONLY / RIGHT_ONLY / BOTH
    straight_track_side = SearchSide.LEFT_ONLY
    # 由 straight_track_side 一次性派生（循环外算一次）
    if straight_track_side == SearchSide.LEFT_ONLY:
        straight_side = SearchSide.LEFT_ONLY
        straight_mode = MidLineMode.LEFT_OFFSET
    elif straight_track_side == SearchSide.RIGHT_ONLY:
        straight_side = SearchSide.RIGHT_ONLY
        straight_mode = MidLineMode.RIGHT_OFFSET
    else:
        straight_side = SearchSide.BOTH
        straight_mode = MidLineMode.MID_AVG

    # 状态机变量
    state = ProcessState.IDLE

    try:
        while True:
            if paused:
                key = cv2.waitKey(50) & 0xFF
                if key == ord(" "):
                    paused = False
                    prev_t = None
                    logging.info("Video resumed")
                elif key == ord("q"):
                    logging.info("User quit")
                    break
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                logging.info("Video ended or read failed")
                break

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
                    logging.info("State: IDLE -> STRAIGHT_TRACKING (straight received)")
                elif right_received:
                    state = ProcessState.RIGHT_TRACKING
                    logging.info("State: IDLE -> RIGHT_TRACKING (right received)")
                elif left_received:
                    state = ProcessState.LEFT_TRACKING
                    logging.info("State: IDLE -> LEFT_TRACKING (left received)")

            # --- 非 IDLE 状态才执行图像处理 ---
            else:
                # 预处理
                binary_img = imgprocess.preprocess(frame)

                if state in (ProcessState.RIGHT_TRACKING, ProcessState.LEFT_TRACKING):
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

                    canvas = imgprocess.draw_line(canvas, state=state)

                    cv2.imshow("binary", binary_img)
                    cv2.imshow("processed_img", canvas)
                else:
                    # STRAIGHT_TRACKING
                    imgprocess.set_mid_line_mode(straight_mode)
                    canvas = imgprocess.return_frame()
                    imgprocess.get_side_line_task_2(
                        binary_img,
                        canvas,
                        is_draw=True,
                        find_corner=False,
                        side=straight_side,
                    )
                    # 多项式拟合
                    imgprocess.fit_polynomial()

                    canvas = imgprocess.draw_line(canvas, fps, state)

                    cv2.imshow("binary", binary_img)
                    cv2.imshow("processed_img", canvas)

            # 按键控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # q键退出
                logging.info("User quit")
                break
            elif key == ord(" "):  # 空格键暂停
                paused = True
                logging.info("Video paused")
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


def main_perspective():
    # FPS 计算：使用高精度计时器，逐帧更新 imgprocess.fps
    prev_t = None  # 上一帧时间戳（time.perf_counter）
    fps = 0.0
    dt = 0.0
    j = 0.0

    imgprocess = ImageProcess()

    try:
        ip = "127.0.0.1"
        port = 12345
        img_sender = ImageSender(ip, port)

        # 连接服务器，开启发送线程
        img_sender.connect()

        img_sender.start_sending(3)

        cap = CameraCapture(0)  # 0表示默认摄像头
        if not cap.is_opened():
            logging.error("Cannot open camera.")
            return

        while True:
            frame = cap.get_picture()
            # frame = cap.correct_img(frame)
            # frame = cv2.flip(frame, 1)  # 水平翻转

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

            canvas = imgprocess.return_frame()
            perspective_frame = imgprocess.perspective_transform(frame, matrix=None)
            # cv2.imshow("perspective", perspective_frame)
            img_sender.enqueue_image(perspective_frame, img_id=0, img_name="perspect")

            binary_img = imgprocess.preprocess(perspective_frame)
            # cv2.imshow("binary", binary_img)
            img_sender.enqueue_image(binary_img, img_id=0, img_name="binary")

            # 获取边线（传入canvas用于绘制调试信息）
            imgprocess.get_side_line_task_2(
                binary_img,
                canvas,
                is_draw=True,
                find_corner=False,
            )
            imgprocess.fit_polynomial()

            canvas = imgprocess.draw_line(canvas, fps)

            # cv2.imshow("canvas", canvas)
            img_sender.enqueue_image(canvas, img_id=0, img_name="canvas")

    except KeyboardInterrupt:
        logging.info("Interrupted by user , start exit...")
    except Exception as e:
        logging.error(f"Error occurred in main_perspective: {e}")
    finally:
        cap.close()
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

    # STRAIGHT_TRACKING 走哪一边：LEFT_ONLY / RIGHT_ONLY / BOTH
    straight_track_side = SearchSide.LEFT_ONLY
    # 由 straight_track_side 一次性派生（循环外算一次）
    if straight_track_side == SearchSide.LEFT_ONLY:
        straight_side = SearchSide.LEFT_ONLY
        straight_mode = MidLineMode.LEFT_OFFSET
    elif straight_track_side == SearchSide.RIGHT_ONLY:
        straight_side = SearchSide.RIGHT_ONLY
        straight_mode = MidLineMode.RIGHT_OFFSET
    else:
        straight_side = SearchSide.BOTH
        straight_mode = MidLineMode.MID_AVG

    # 状态机变量
    state = ProcessState.IDLE

    try:
        # 连接服务器，开启发送线程
        # img_sender.connect()

        # img_sender.start_sending(2)

        cap = CameraCapture(0)  # 0表示默认摄像头
        if not cap.is_opened():
            logging.error("Cannot open camera.")
            return

        while True:
            frame = cap.get_picture()
            # frame = cap.correct_img(frame)
            # frame = cv2.flip(frame, 1)  # 水平翻转

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
                binary_img = imgprocess.preprocess(frame)

                if state in (ProcessState.RIGHT_TRACKING, ProcessState.LEFT_TRACKING):
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

                    canvas = imgprocess.draw_line(canvas, fps, state)

                    cv2.imshow("binary", binary_img)
                    cv2.imshow("processed_img", canvas)
                else:
                    # STRAIGHT_TRACKING
                    imgprocess.set_mid_line_mode(straight_mode)
                    canvas = imgprocess.return_frame()
                    imgprocess.get_side_line_task_2(
                        binary_img,
                        canvas,
                        is_draw=True,
                        find_corner=False,
                        side=straight_side,
                    )
                    # 多项式拟合
                    imgprocess.fit_polynomial()

                    canvas = imgprocess.draw_line(canvas, fps, state)

                    cv2.imshow("binary", binary_img)
                    cv2.imshow("processed_img", canvas)

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
    # main_video()
    main()
