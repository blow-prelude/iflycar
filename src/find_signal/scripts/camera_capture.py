import logging
import os
import time

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class CameraCapture:
    def __init__(self, index=0, width=640, height=480) -> None:
        """初始化摄像头

        Args:
            index: 摄像头索引，默认使用 CameraConfig.INDEX
            width: 分辨率宽度，默认使用 CameraConfig.WIDTH
            height: 分辨率高度，默认使用 CameraConfig.HEIGHT
        """

        self.cap = cv2.VideoCapture(index)
        # self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            logging.error("Cannot open camera.")
            raise RuntimeError(f"Failed to open camera at index {index}")

        self.mtx = np.array([])  # 内参数矩阵
        self.dist = np.array([])  # 畸变系数

    def get_picture(self):
        """获取一帧图片"""
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Camera is not opened.")

        ret, frame = self.cap.read()
        if not ret:
            logging.error("Failed to grab frame")
            raise RuntimeError("Failed to grab frame")
        frame_1 = cv2.flip(frame, 1)  # 水平翻转
        return frame_1

    def save_picture(self, frame, filename):
        """保存图片到指定文件"""
        if frame is not None:
            cv2.imwrite(filename, frame)
            logging.info(f"Saved picture to {filename}")
        else:
            logging.warning("No frame to save.")

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

    def calibration(self, sample_dir):
        # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )  # 阈值
        # 棋盘格模板规格
        w = 9  # 10 - 1
        h = 9  # 10  - 1

        # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
        objp = np.zeros((w * h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        objp = objp * 18.1  # 18.1 mm

        objpoints = []  # 在世界坐标系中的三维点
        imgpoints = []  # 在图像平面的二维点

        images = [
            os.path.join(sample_dir, f)
            for f in os.listdir(sample_dir)
            if f.endswith(".png")
        ]

        i = 0

        for image in images:
            try:
                img = cv2.imread(image)
                if img is None:
                    raise ValueError("Failed to read image")  # TODO 什么类型呢
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                ih, iw = img.shape[:2]
                # 找到棋盘格角点
                ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
                # 如果找到足够点对，将其存储起来
                if ret is True:
                    logging.info(f"calibrating:{i}")
                    i = i + 1
                    # 在原角点的基础上寻找亚像素角点
                    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    # 追加进入世界三维点和平面二维点中
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    # 将角点在图像上显示
                    cv2.drawChessboardCorners(img, (w, h), corners, ret)
                    cv2.namedWindow("findCorners", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("findCorners", 640, 480)
                    cv2.imshow("findCorners", img)
                    cv2.waitKey(200)
            except Exception as e:
                logging.error(f"error when calibrating:{e}")
        cv2.destroyAllWindows()

        # 标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        self.mtx = np.array(mtx, dtype=np.float32)
        self.dist = np.array(dist, dtype=np.float32)

        logging.info(f"ret:{ret}")  # 重投影误差，越小说明标定越准
        logging.info(f"mtx:\n{mtx}")  # 内参数矩阵，包括焦距和光心
        logging.info(f"dist畸变值:\n{dist}")  # 畸变系数，包括径向畸变和切向畸变

        # 根据畸变参数，计算一个去畸变后的最优内参矩阵
        # roi： 去畸变后可剪掉黑边
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (ih, iw), 0, (ih, iw)
        )

    def correct_img(self, img):
        h, w = img.shape[:2]
        try:
            if len(self.mtx) > 0 and len(self.dist) > 0:
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                    self.mtx, self.dist, (h, w), 0, (h, w)
                )

            # 生成去畸变映射表，并应用映射表将像素重新映射
            mapx, mapy = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, None, newcameramtx, (w, h), 5
            )
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            return dst
        except Exception as e:
            logging.error(f"error when correct img with mtx:{e}")
            return img

    def check_camera_capabilities(self):
        """检测摄像头支持哪些功能"""

        capabilities = {
            "分辨率调整": "CAP_PROP_FRAME_WIDTH",
            "帧率调整": "CAP_PROP_FPS",
            "亮度": "CAP_PROP_BRIGHTNESS",
            "对比度": "CAP_PROP_CONTRAST",
            "饱和度": "CAP_PROP_SATURATION",
            "色调": "CAP_PROP_HUE",
            "增益": "CAP_PROP_GAIN",
            "曝光": "CAP_PROP_EXPOSURE",
            "自动聚焦": "CAP_PROP_AUTOFOCUS",
            "手动聚焦": "CAP_PROP_FOCUS",
            "自动白平衡": "CAP_PROP_AUTO_WB",
            "白平衡色温": "CAP_PROP_WB_TEMPERATURE",
            "缩放": "CAP_PROP_ZOOM",
            "锐度": "CAP_PROP_SHARPNESS",
            "伽马": "CAP_PROP_GAMMA",
            "背光补偿": "CAP_PROP_BACKLIGHT",
        }

        logging.info("\n摄像头功能检测：")

        for feature_name, prop_name in capabilities.items():
            prop = getattr(cv2, prop_name)

            # 尝试获取当前值
            current_value = self.cap.get(prop)

            # 尝试设置一个测试值
            test_value = current_value + 1 if current_value >= 0 else 100
            set_success = self.cap.set(prop, test_value)
            new_value = self.cap.get(prop)

            # 判断是否支持
            if set_success and new_value != current_value:
                status = "✓ 支持"
                logging.info(
                    f"{feature_name:<15}: {status}  (当前值: {current_value:.2f})"
                )
                # 恢复原值
                self.cap.set(prop, current_value)
            else:
                status = "✗ 不支持"
                logging.info(f"{feature_name:<15}: {status}")

        logging.info("=" * 60)
        self.close()


if __name__ == "__main__":
    # 获取脚本所在目录的上级目录的pictures子目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pictures_dir = os.path.abspath(os.path.join(script_dir, "..", "pictures"))
    os.makedirs(pictures_dir, exist_ok=True)

    i = 0
    j = 0
    curr_t, prev_t, dt = None, None, 0.0
    cap = CameraCapture(0)
    try:
        while True:
            curr_t = time.time()
            if prev_t is not None:
                dt += curr_t - prev_t
                j += 1
                # 每10轮计算一次
                if j % 10 == 0 and dt > 1e-6:
                    fps = 1 / dt * 10
                    logging.info(f"fps:{fps}")
                    dt = 0.0
            prev_t = curr_t

            frame = cap.get_picture()

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                filename = os.path.join(pictures_dir, f"sample{i}.png")
                cap.save_picture(frame, filename)
                i += 1

    except KeyboardInterrupt:
        logging.warning("interrupted by user, exiting...")

    except Exception as e:
        logging.error(f"error in main loop: {e}")

    finally:
        cap.close()
        cv2.destroyAllWindows()
