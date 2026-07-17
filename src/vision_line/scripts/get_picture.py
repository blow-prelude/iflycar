import logging
import os
from datetime import datetime

import cv2
from camera_capture import CameraCapture

if __name__ == "__main__":
    with CameraCapture() as cap:
        frame = cap.get_picture()
        frame = cap.correct_img(frame)
        frame = cv2.flip(frame, 1)  # 水平翻转，得到正常视角（左转时不翻转，保持原视角）
        frame = cv2.resize(frame, (320, 240))
        if frame is not None:
            # 通过相对路径保存到上一级目录的 pictures/ 下
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.abspath(os.path.join(script_dir, "..", "pictures"))
            os.makedirs(save_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_image_{timestamp}.jpg"
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, frame)
            logging.info(f"Image saved to {save_path}")

            cv2.imshow("Captured Image", frame)
            cv2.waitKey(0)
        else:
            logging.error("Failed to capture image.")
