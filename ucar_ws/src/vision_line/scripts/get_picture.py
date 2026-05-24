import logging
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
            # 保存图片到当前路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_image_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            logging.info(f"Image saved to {filename}")

            cv2.imshow("Captured Image", frame)
            cv2.waitKey(0)
        else:
            logging.error("Failed to capture image.")
