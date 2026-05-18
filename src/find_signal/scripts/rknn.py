import logging
import os
import queue
import threading
import time

import cv2
import numpy as np
from camera_capture import CameraCapture
from rknn_det import TextDetector
from rknn_rec import TextRecognizer
from rknnlite.api import RKNNLite

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

IMG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.jpg")
DET_INPUT_SHAPE = [480, 480]
REC_INPUT_SHAPE = [48, 320]


class img_processor:
    def __init__(self) -> None:
        self.cur_fps = 0.0

    def queue_img(self, img, input_queue):
        try:
            input_queue.put(img, block=False)
        except queue.Full:
            # 队列满时丢弃最旧的图像
            try:
                input_queue.get_nowait()
                input_queue.put(img, block=False)
            except queue.Empty:
                pass

    def update_fps(self, fps):
        if fps > 0.0:
            self.cur_fps = fps

    def draw_fps(self, img):
        if self.cur_fps > 0.0:
            cv2.putText(
                img,
                "FPS: {:.1f}".format(self.cur_fps),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )


def post_process(output):
    pass


def crop_text_region(img, output):
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
        logging.debug(f"Largest box: {max_box}, size: {x_max - x_min}x{y_max - y_min}")
    else:
        cropped = None
    return cropped


def inference_worker(
    det_model, rec_model, input_queue, det_output_queue, rec_output_queue
):
    while True:
        try:
            img = input_queue.get(timeout=1)  # 等待图像输入
            time1 = time.perf_counter()
            det_output = det_model.run(img)
            det_output_queue.put(det_output)  # 放入检测结果
            logging.info(f"det inference time: {time.perf_counter() - time1:.4f} s")

            cropped = crop_text_region(img, det_output)
            if cropped is not None:
                cropped = cv2.resize(cropped, (REC_INPUT_SHAPE[1], REC_INPUT_SHAPE[0]))
                time1 = time.perf_counter()
                rec_output = rec_model.run(cropped)
                logging.info(f"rec inference time: {time.perf_counter() - time1:.4f} s")
                rec_output_queue.put(rec_output)
        except queue.Empty:
            continue  # 没有图像输入，继续等待


def main():
    input_queue = queue.Queue(maxsize=10)  # 图像队列
    det_output_queue, rec_output_queue = (
        queue.Queue(maxsize=10),
        queue.Queue(maxsize=10),
    )  # 结果队列

    cur_t = 0.0
    pre_t = 0.0
    elapsed = 0.0
    frame_count = 0

    try:
        det_model0 = TextDetector(target="rk3588", device_id=RKNNLite.NPU_CORE_0_1)
        rec_model0 = TextRecognizer(target="rk3588", device_id=RKNNLite.NPU_CORE_0_1)
        det_model1 = TextDetector(target="rk3588", device_id=RKNNLite.NPU_CORE_2)
        rec_model1 = TextRecognizer(target="rk3588", device_id=RKNNLite.NPU_CORE_2)
        # det_model2 = TextDetector(target="rk3588", device_id=RKNNLite.NPU_CORE_2)
        # rec_model2 = TextRecognizer(target="rk3588", device_id=RKNNLite.NPU_CORE_2)
        logging.info("Models initialized successfully")

        cap = CameraCapture(0)
        img_proc = img_processor()

        det_models = [det_model0, det_model1]
        rec_models = [rec_model0, rec_model1]
        workers = []

    except Exception as e:
        logging.error(f"Error init: {e}")
        exit(1)

    try:
        for det_model, rec_model in zip(det_models, rec_models):
            worker = threading.Thread(
                target=inference_worker,
                args=(
                    det_model,
                    rec_model,
                    input_queue,
                    det_output_queue,
                    rec_output_queue,
                ),
            )
            worker.daemon = True
            worker.start()
            workers.append(worker)

        while True:
            frame = cap.get_picture()
            frame = cap.correct_img(frame)
            frame = cv2.flip(frame, 1)
            # cv2.imshow("Camera", frame)

            frame = cv2.resize(frame, (DET_INPUT_SHAPE[1], DET_INPUT_SHAPE[0]))
            canvas = frame.copy()

            # cv2.imshow("before detection", frame)
            img_proc.queue_img(frame, input_queue)

            try:
                cur_t = time.perf_counter()
                elapsed += cur_t - pre_t
                frame_count += 1
                pre_t = cur_t

                if frame_count % 10 == 0:
                    fps = frame_count / elapsed if elapsed > 0 else 0.0
                    # logging.info(f"Current FPS: {fps:.2f}")
                    img_proc.update_fps(fps)
                    elapsed = 0
                    frame_count = 0

                det_output = det_output_queue.get(timeout=0.15)  # 获取检测结果

                for box in det_output:
                    box = np.array(box).astype(np.int32)
                    cv2.polylines(canvas, [box], True, (0, 255, 0), 2)

                img_proc.draw_fps(canvas)
                cv2.imshow("Detection Results", canvas)
                cv2.waitKey(1)

                rec_output = rec_output_queue.get(timeout=0.1)  # 获取识别结果
                logging.info(f"Recognition result: {rec_output}")

            except queue.Empty:
                img_proc.draw_fps(canvas)
                cv2.imshow("Detection Results", canvas)
                cv2.waitKey(1)

            # print("Detection results:", output.tolist())

    except KeyboardInterrupt:
        print("Exiting...")

    except Exception as e:
        print(f"Error during processing: {e}")

    finally:
        cv2.destroyAllWindows()
        cap.close()
        for det_model in det_models:
            det_model.release()
        for rec_model in rec_models:
            rec_model.release()


if __name__ == "__main__":
    main()
