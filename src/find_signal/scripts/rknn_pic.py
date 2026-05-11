import os
import time

import cv2
import numpy as np
from rknn_det import TextDetector
from rknn_rec import TextRecognizer
from rknnlite.api import RKNNLite

IMG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.jpg")
DET_INPUT_SHAPE = [480, 480]
REC_INPUT_SHAPE = [48, 320]

if __name__ == "__main__":
    try:
        det_model = TextDetector(target="rk3588", device_id=RKNNLite.NPU_CORE_0_1)
        rec_model = TextRecognizer(target="rk3588", device_id=RKNNLite.NPU_CORE_0_1)

    except Exception as e:
        print(f"Error initializing models: {e}")
        exit(1)

    try:
        img = cv2.imread(IMG_PATH)
        img = cv2.resize(img, (DET_INPUT_SHAPE[1], DET_INPUT_SHAPE[0]))

        time1 = time.perf_counter()
        output = det_model.run(img)
        print(f"det inference time: {time.perf_counter() - time1:.4f} s")

        for box in output:
            box = np.array(box).astype(np.int32)
            cv2.polylines(img, [box], True, (0, 255, 0), 2)

        print("Detection results:", output.tolist())

        # Find the largest bounding box
        max_area = 0
        max_box = None
        for box in output:
            box = np.array(box)
            x_coords = box[:, 0]
            y_coords = box[:, 1]
            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            y_min, y_max = int(y_coords.min()), int(y_coords.max())
            area = (x_max - x_min) * (y_max - y_min)
            if area > max_area:
                max_area = area
                max_box = (x_min, y_min, x_max, y_max)

        # Crop the largest box from original image
        if max_box:
            x_min, y_min, x_max, y_max = max_box
            cropped = img[y_min:y_max, x_min:x_max]
            print(f"Largest box: {max_box}, size: {x_max - x_min}x{y_max - y_min}")
        else:
            cropped = None
            print("No text detected")
        if cropped is None:
            exit(0)

        cropped = cv2.resize(cropped, (REC_INPUT_SHAPE[1], REC_INPUT_SHAPE[0]))
        time1 = time.perf_counter()
        rec_output = rec_model.run(cropped)
        print(f"rec inference time: {time.perf_counter() - time1:.4f} s")
        print("Recognition results:", rec_output)

    finally:
        cv2.destroyAllWindows()
        det_model.release()
        rec_model.release()
