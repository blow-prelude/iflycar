import cv2
import numpy as np

cap = cv2.VideoCapture(0)
# 定义保存的文件名以及编码格式
output_file = "output1.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 也可以选择其他格式，如 'MJPG', 'mp4v', 等

mtx = np.array([[408.41864465257, 0, 309.7039339414456],[0, 406.0509463667899, 272.3979542024044],[0, 0, 1]], dtype=np.float32)
dist = np.array([-0.2816538317937755, 0.06677467622710392, -0.007883381249276117, 0.000701423277555337, 0], dtype=np.float32)



ret, frame = cap.read()

# 初始化VideoWriter对象
video_writer = cv2.VideoWriter(output_file, fourcc, 30, (frame.shape[1], frame.shape[0]))

while True:
    ret, frame = cap.read()
    undistort = cv2.undistort(frame, mtx, dist, None, mtx)
    cv2.imshow('frame', undistort)
    video_writer.write(undistort)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
video_writer.release()
print(f"视频已保存至 {output_file}")
