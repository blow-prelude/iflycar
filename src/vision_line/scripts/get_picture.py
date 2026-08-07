import logging
import os
from datetime import datetime

import cv2
from camera_capture import CameraCapture

if __name__ == "__main__":
    with CameraCapture() as cap:
        # 通过相对路径保存到上一级目录的 pictures/ 下
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.abspath(os.path.join(script_dir, "..", "pictures"))
        os.makedirs(save_dir, exist_ok=True)

        while True:
            frame = cap.get_picture()
            cv2.imshow("Original Frame", frame)
            # frame = cv2.resize(frame, (320, 240))  # Resize for better visibility
            frame = cap.correct_img(frame)
            frame = cv2.flip(
                frame, 1
            )  # 水平翻转，得到正常视角（左转时不翻转，保持原视角）
            frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)

            cv2.imshow("Captured Image", frame)
            k = cv2.waitKey(1)
            if k == ord(" ") and frame is not None:  # 空格键保存图片
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captured_image_{timestamp}.jpg"
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, frame)
                logging.info(f"Image saved to {save_path}")

    cv2.destroyAllWindows()
