import os
import time

import cv2
import numpy as np
from camera_capture import CameraCapture
from rknn_det import TextDetector
from rknn_rec import TextRecognizer
from rknnlite.api import RKNNLite

IMG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.jpg")
DET_INPUT_SHAPE = [480, 480]
REC_INPUT_SHAPE = [48, 320]


def main():
    try:
        det_model = TextDetector(target="rk3588", device_id=RKNNLite.NPU_CORE_0_1)
        rec_model = TextRecognizer(target="rk3588", device_id=RKNNLite.NPU_CORE_0_1)
        print("Models initialized successfully")

    except Exception as e:
        print(f"Error initializing models: {e}")
        exit(1)

    try:
        cap = CameraCapture(0)
        while True:
            frame = cap.get_picture()
            frame = cap.correct_img(frame)
            frame = cv2.flip(frame, 1)
            cv2.imshow("Camera", frame)

            frame = cv2.resize(frame, (DET_INPUT_SHAPE[1], DET_INPUT_SHAPE[0]))
            # cv2.imshow("before detection", frame)

            time1 = time.perf_counter()
            output = det_model.run(frame)
            print(f"det inference time: {time.perf_counter() - time1:.4f} s")

            for box in output:
                box = np.array(box).astype(np.int32)
                cv2.polylines(frame, [box], True, (0, 255, 0), 2)

            cv2.imshow("Detection Results", frame)
            # print("Detection results:", output.tolist())

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
                cropped = frame[y_min:y_max, x_min:x_max]
                print(f"Largest box: {max_box}, size: {x_max - x_min}x{y_max - y_min}")
            else:
                cropped = None
                # print("No text detected")

            if cropped is None:
                continue

            cropped = cv2.resize(cropped, (REC_INPUT_SHAPE[1], REC_INPUT_SHAPE[0]))
            time1 = time.perf_counter()
            rec_output = rec_model.run(cropped)
            print(f"rec inference time: {time.perf_counter() - time1:.4f} s")
            print("Recognition results:", rec_output)

            cv2.waitKey(1)

    except KeyboardInterrupt:
        print("Exiting...")

    except Exception as e:
        print(f"Error during processing: {e}")

    finally:
        cv2.destroyAllWindows()
        cap.close()
        det_model.release()
        rec_model.release()


if __name__ == "__main__":
    main()
