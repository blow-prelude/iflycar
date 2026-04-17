import logging
import os
import time

import cv2

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

    def get_picture(self):
        """获取一帧图片并做预处理"""
        if self.cap is None or not self.cap.isOpened():
            logging.error("Camera is not opened.")
            return None
        ret, frame = self.cap.read()
        if not ret:
            logging.error("Failed to grab frame")
            return None
        return frame

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


if __name__ == "__main__":
    # 获取脚本所在目录的上级目录的pictures子目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pictures_dir = os.path.abspath(os.path.join(script_dir, "..", "pictures"))
    os.makedirs(pictures_dir, exist_ok=True)

    i = 0
    j = 0
    curr_t, prev_t, dt = None, None, 0.0
    try:
        cap = CameraCapture(0)
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
        logging.warning("interrupt bu user, exiting...")

    finally:
        cap.close()
        cv2.destroyAllWindows()
