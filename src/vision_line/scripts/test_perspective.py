import cv2
import numpy as np
from camera_capture import CameraCapture

# 通过 manual_calibration.py 标定得到的透视变换矩阵
TRANSFORMATION_MATRIX = np.array(
    [
        [-0.604762, -1.374288, 268.961830],
        [0.045697, -2.514863, 361.149734],
        [0.000223, -0.008647, 1.000000],
    ],
    dtype=np.float32,
)


if __name__ == "__main__":
    with CameraCapture() as cap:
        while True:
            frame = cap.get_picture()
            frame_1 = cap.correct_img(frame)
            frame = cv2.flip(
                frame_1, 1
            )  # 水平翻转，得到正常视角（左转时不翻转，保持原视角）
            frame = cv2.resize(frame, (320, 240))  # 调整图像大小以适应窗口显示

            warped = cv2.warpPerspective(
                frame,
                TRANSFORMATION_MATRIX,
                (frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_LINEAR,
            )

            cv2.imshow("Original Frame", frame)
            cv2.imshow("Perspective View", warped)
            cv2.waitKey(1)

        cv2.destroyAllWindows()
