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

        self.cap = cv2.VideoCapture(idx)
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
        if hasattr(self, 'cap'):
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
        self.mid_line = []
        self.fit_mid_line = []

    def preprocess(self):
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
            binary = cv2.threshold(grey, 100, 255, cv2.THRESH_BINARY)[1]
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
        up_ratio = 0.4
        down_ratio = 0.9
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
                                < 3
                            ):
                                self.right_line.append((i, j))

                # 如果左右都找到的边线，则计算中线
                if left_f == 1 and right_f == 1:
                    mid_x = (
                        self.left_line[len(self.left_line) - 1][0]
                        + self.right_line[len(self.right_line) - 1][0]
                    ) // 2
                    self.mid_line.append((mid_x, j))
                    left_f = 0
                    right_f = 0
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
                f"length of left_line: {len(self.left_line)} , lenth of right_line: {len(self.right_line)}"
            )
            if self.frame is None:
                raise ValueError("frame is None")
            img = self.frame.copy()
            for i in range(len(self.left_line)):
                cv2.circle(img, self.left_line[i], 2, (0, 0, 255), -1)
            for i in range(len(self.right_line)):
                cv2.circle(img, self.right_line[i], 2, (0, 255, 0), -1)
            for i in range(len(self.mid_line)):
                cv2.circle(img, self.mid_line[i], 2, (255, 0, 0), -1)
            for i in range(len(self.fit_mid_line)):
                cv2.circle(img, self.fit_mid_line[i], 2, (255, 255, 0), -1)
            return img
        except Exception as e:
            logging.error(f"Error occurred during drawing lines: {e}")
            return self.frame


def main():
    imgprocess = ImageProcess()
    while True:
        try:
            img = imgprocess.preprocess()
            cv2.imshow("img", img)
            imgprocess.get_side_line(img)
            imgprocess.fit_polynomial()
            canvas = imgprocess.draw_line()
            cv2.imshow("processed_img", canvas)
            cv2.waitkey(1)
        except KeyboardInterrupt:
            logging.info("shutdown by user")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
