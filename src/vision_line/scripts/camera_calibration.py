import os

import cv2
from camera_capture import CameraCapture

cap = CameraCapture(0)

# 加载samples文件夹下所有的png图像
script_dir = os.path.dirname(os.path.abspath(__file__))
samples_dir = os.path.abspath(os.path.join(script_dir, "..", "samples"))

cap.calibration(samples_dir)

try:
    while True:
        frame = cap.get_picture()
        dst = cap.correct_img(frame)


        cv2.imshow("frame", frame)
        cv2.imshow('dst',dst)
        if cv2.waitKey(1) & 0xFF == ord("q"): 
            break
except KeyboardInterrupt:
    print("\nInterrupted by user, exiting...")
except Exception as e:
    print("Error:", e)

finally:
    cap.close()
    cv2.destroyAllWindows()
