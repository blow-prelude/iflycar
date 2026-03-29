import logging

import cv2
import numpy as np


class CameraConfig:
    INDEX = 0
    WIDTH = 640
    HEIGHT = 480


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class CameraCapture:
    def __init__(self, index=None, width=None, height=None) -> None:
        """初始化摄像头

        Args:
            index: 摄像头索引，默认使用 CameraConfig.INDEX
            width: 分辨率宽度，默认使用 CameraConfig.WIDTH
            height: 分辨率高度，默认使用 CameraConfig.HEIGHT
        """
        idx = index if index is not None else CameraConfig.INDEX
        w = width if width is not None else CameraConfig.WIDTH
        h = height if height is not None else CameraConfig.HEIGHT

        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        if not self.cap.isOpened():
            logging.error("Cannot open camera.")
            raise RuntimeError(f"Failed to open camera at index {idx}")

    def get_picture(self):
        """获取一帧图片并做预处理"""
        ret, frame = self.cap.read()
        if not ret:
            logging.error("Failed to grab frame")
            return None
        return frame

    def is_opened(self):
        """检查摄像头是否已打开"""
        return self.cap is not None and self.cap.isOpened()

    def close(self):
        """关闭摄像头"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """上下文管理器出口，自动释放资源"""
        self.close()
        return False

    def __del__(self):
        """析构函数，对象销毁时自动释放资源"""
        self.close()


class ImageProcess:
    def __init__(self, img_path=None):
        if img_path is None:
            self.cap = CameraCapture(
                CameraConfig.INDEX, CameraConfig.WIDTH, CameraConfig.HEIGHT
            )
        else:
            self.cap = None
            self.img_path = img_path
        self.frame = None
        self.left_line = []
        self.right_line = []
        self.supple_left_line = []
        self.supple_right_line = []
        self.mid_line = []
        self.fit_mid_line = []

    def preprocess(self):
        """做预处理，得到二值化的图像"""
        try:
            if self.cap is None:
                self.frame = cv2.imread(self.img_path)
            else:
                self.frame = self.cap.get_picture()
            if self.frame is not None and (
                self.frame.shape[0] >= 640 or self.frame.shape[1] >= 480
            ):
                # 将图片按比例缩小，使宽和高都不超过640和480
                h, w = self.frame.shape[:2]
                scale = min(640 / h, 480 / w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                self.frame = cv2.resize(
                    self.frame, (new_w, new_h), interpolation=cv2.INTER_AREA
                )

            grey = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            binary = cv2.threshold(grey, 180, 255, cv2.THRESH_BINARY)[1]
            # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            # erode = cv2.erode(binary, kernel, iterations=2)  # 用腐消除图像中较亮的区域
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            close = cv2.morphologyEx(
                binary, cv2.MORPH_CLOSE, kernel, iterations=3
            )  # 用闭运算消除图像中较暗的区域
            return close
        except Exception as e:
            logging.error(f"Error occurred during image processing: {e}")

    def get_side_line(self, img):
        """从图像的中线往两边搜索，获取赛道边线"""
        mid_x = img.shape[1] // 2
        # 从图像下方（靠近车辆）开始搜索
        up_ratio = 0.55
        down_ratio = 0.95
        try:
            for j in range(
                int(img.shape[0] * down_ratio), int(img.shape[0] * up_ratio), -1
            ):
                left_f, right_f = 0, 0
                # 左侧赛道线
                for i in range(mid_x, -1, -1):
                    if i <= 1:
                        break
                    if img[j, i] != 0 and img[j, i - 1] == 0:
                        logging.debug(
                            f"find left line at {i}, {j} , value : {img[j, i]}"
                        )
                        left_f = 1
                        # 如果列表为空，则直接添加从黑到白的跳变点；否则要判断是否和上一个边界点连续
                        if len(self.left_line) == 0:
                            self.left_line.append((i, j))
                        else:
                            if (
                                abs(self.left_line[len(self.left_line) - 1][0] - i) < 10
                                and abs(self.left_line[len(self.left_line) - 1][1] - j)
                                < 5
                            ):
                                self.left_line.append((i, j))
                            else:
                                logging.debug(
                                    f"jump too far at {i}, {j} , last point : {self.left_line[len(self.left_line) - 1]}"
                                )

                # 右侧赛道线
                for i in range(mid_x, img.shape[1], 1):
                    if i >= img.shape[1] - 1:
                        break
                    if img[j, i] != 0 and img[j, i + 1] == 0:
                        logging.debug(
                            f"find right line at {i}, {j} , value : {img[j, i]}"
                        )
                        right_f = 1
                        if len(self.right_line) == 0:
                            self.right_line.append((i, j))
                        else:
                            if (
                                abs(self.right_line[len(self.right_line) - 1][0] - i)
                                < 10
                                and abs(
                                    self.right_line[len(self.right_line) - 1][1] - j
                                )
                                < 5
                            ):
                                self.right_line.append((i, j))
                            else:
                                logging.debug(
                                    f"jump too far at {i}, {j} , last point : {self.left_line[len(self.left_line) - 1]}"
                                )

                if left_f == 1 and right_f == 1:
                    # 线性补值：对于间隔较大的两个像素点，计算斜率并填充中间点
                    left_before_supple = len(self.left_line)
                    right_before_supple = len(self.right_line)

                    # 对左边线进行线性插值补充
                    if left_before_supple >= 2:
                        self.supple_left_line = self.left_line.copy()
                        i = 0
                        while i < len(self.supple_left_line) - 1:
                            x1, y1 = self.supple_left_line[i]
                            x2, y2 = self.supple_left_line[i + 1]
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
                                        self.supple_left_line.insert(
                                            i + t, (new_x, new_y)
                                        )

                                i += num_points + 1
                            else:
                                i += 1
                    else:
                        self.supple_left_line = self.left_line.copy()

                    # 对右边线进行线性插值补充
                    if right_before_supple >= 2:
                        self.supple_right_line = self.right_line.copy()
                        i = 0
                        while i < len(self.supple_right_line) - 1:
                            x1, y1 = self.supple_right_line[i]
                            x2, y2 = self.supple_right_line[i + 1]
                            dx = abs(x2 - x1)
                            dy = abs(y2 - y1)

                            # 如果两点间距离过大，进行线性插值
                            if dx > 10 or dy > 5:
                                # 计算需要插入的点数量
                                num_points = max(dx // 5, dy // 3, 2)

                                # 在两点之间进行线性插值
                                for t in range(1, num_points + 1):
                                    ratio = t / (num_points + 1)
                                    new_x = int(x1 + (x2 - x1) * ratio)
                                    new_y = int(y1 + (y2 - y1) * ratio)
                                    if min(y1, y2) < new_y < max(y1, y2):
                                        self.supple_right_line.insert(
                                            i + t, (new_x, new_y)
                                        )

                                i += num_points + 1
                            else:
                                i += 1
                    else:
                        self.supple_right_line = self.right_line.copy()

                    # 将图像边界填充为白色，防止计算中线时越界
                    # 在列表最前端加上图像边界的像素点，使得赛道一直延伸到底端
                    if len(self.supple_left_line) > 0:
                        bottom_point = self.supple_left_line[0]
                        bottom_y = bottom_point[1]
                        # 从图像底部（img.shape[0]-1）向上延伸到第一个点
                        temp_left_line = []
                        for j1 in range(img.shape[0] - 1, bottom_y, -2):
                            # 左边线延伸到图像左边界 (x=0)
                            temp_left_line.append((0, j1))
                        self.supple_left_line = temp_left_line + self.supple_left_line

                    if len(self.supple_right_line) > 0:
                        bottom_point = self.supple_right_line[0]
                        bottom_y = bottom_point[1]
                        # 从图像底部向上延伸到第一个点
                        temp_right_line = []
                        for j1 in range(img.shape[0] - 1, bottom_y, -2):
                            # 右边线延伸到图像右边界
                            temp_right_line.append((img.shape[1] - 1, j1))
                        self.supple_right_line = (
                            temp_right_line + self.supple_right_line
                        )

                    left_f = 0
                    right_f = 0

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

            logging.info(
                f"Polynomial fitting completed: {len(self.fit_mid_line)} points, "
                f"coefficients: {coefficients}"
            )
        except Exception as e:
            logging.error(f"Error occurred during polynomial fitting: {e}")

    def draw_line(self):
        """绘制边线"""
        try:
            logging.info(
                f"length of left_line: {len(self.supple_left_line)} , lenth of right_line: {len(self.supple_right_line)}"
            )
            if self.frame is None:
                raise ValueError("frame is None")
            img = self.frame.copy()
            # 绘制优化后的左边线（红色）
            for i in range(len(self.supple_left_line)):
                cv2.circle(img, self.supple_left_line[i], 2, (0, 0, 255), -1)
            # 绘制优化后的右边线（绿色）
            for i in range(len(self.supple_right_line)):
                cv2.circle(img, self.supple_right_line[i], 2, (0, 255, 0), -1)
            for i in range(len(self.mid_line)):
                cv2.circle(img, self.mid_line[i], 2, (255, 0, 0), -1)
            for i in range(len(self.fit_mid_line)):
                cv2.circle(img, self.fit_mid_line[i], 2, (255, 255, 0), -1)
            return img
        except Exception as e:
            logging.error(f"Error occurred during drawing lines: {e}")
            return self.frame

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
            if white_counts is None:
                raise ValueError("white_counts is None")

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


def main():
    imgprocess = ImageProcess(r"D:\programs\ucar_ws\src\vision_line\pictures\test3.png")
    img = imgprocess.preprocess()
    cv2.imshow("img", img)

    ##  绘制直方图
    # hist_img_lower, white_counts, peaks = imgprocess.plot_column_histogram(
    #     img, 0.6, 0.9
    # )
    # if hist_img_lower is not None:
    #     cv2.imshow("histogram_lower_half", hist_img_lower)
    #     logging.info(f"Detected peaks at x positions: {peaks}")
    #     if len(peaks) >= 2:
    #         left_peak = peaks[0]
    #         right_peak = peaks[-1]
    #         center = (left_peak + right_peak) // 2
    #         logging.info(
    #             f"Left line: {left_peak}, Right line: {right_peak}, Center: {center}"
    #         )

    imgprocess.get_side_line(img)
    imgprocess.fit_polynomial()
    canvas = imgprocess.draw_line()
    cv2.imshow("processed_img", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
