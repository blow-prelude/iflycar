import cv2
import numpy as np

mtx = np.array([[408.41864465257, 0, 309.7039339414456],[0, 406.0509463667899, 272.3979542024044],[0, 0, 1]], dtype=np.float32)
dist = np.array([-0.2816538317937755, 0.06677467622710392, -0.007883381249276117, 0.000701423277555337, 0], dtype=np.float32)

cap = cv2.VideoCapture("/dev/video0")
while True:
    ret, frame = cap.read()

    undistort = cv2.undistort(frame, mtx, dist, None, mtx)

    print(frame.shape)
    cv2.imshow('frame', frame)
    cv2.imshow('undistort', undistort)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()