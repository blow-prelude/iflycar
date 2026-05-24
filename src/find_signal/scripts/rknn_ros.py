import json
import logging
import os
import queue
import threading
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point32
from sensor_msgs.msg import Image
from std_msgs.msg import String
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

ros_image = None
bridge = CvBridge()


def image_callback(data):
    global ros_image
    try:
        ros_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(e)


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

    def letter_box(
        self,
        im,
        new_shape,
        pad_color=(0, 0, 0),
    ):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
        )  # add border

        return im, ratio, (dw, dh)


def get_center_point(box):
    box = np.array(box).reshape(-1, 2)
    x_mid = int((box[:, 0].min() + box[:, 0].max()) / 2)
    y_mid = int((box[:, 1].min() + box[:, 1].max()) / 2)
    return (x_mid, y_mid)


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
            max_box = box

    if max_box is not None:
        x_min = int(max_box[:, 0].min())
        x_max = int(max_box[:, 0].max())
        y_min = int(max_box[:, 1].min())
        y_max = int(max_box[:, 1].max())
        cropped = img[y_min:y_max, x_min:x_max]
        logging.debug(
            f"Largest box: ({x_min}, {y_min}, {x_max}, {y_max}), size: {x_max - x_min}x{y_max - y_min}"
        )
    else:
        cropped = None
    return cropped, max_box


def inference_worker(
    det_model, rec_model, input_queue, det_output_queue, rec_output_queue
):
    while True:
        try:
            img = input_queue.get(timeout=1)  # 等待图像输入
            time1 = time.perf_counter()
            det_output = det_model.run(img)
            logging.debug(f"det inference time: {time.perf_counter() - time1:.4f} s")

            cropped, corners = crop_text_region(img, det_output)
            if corners is not None:
                det_output_queue.put([corners.astype(np.int32)])
            if cropped is not None:
                cropped = cv2.resize(cropped, (REC_INPUT_SHAPE[1], REC_INPUT_SHAPE[0]))
                time1 = time.perf_counter()
                rec_output = rec_model.run(cropped)
                logging.debug(
                    f"rec inference time: {time.perf_counter() - time1:.4f} s"
                )
                rec_output_queue.put(rec_output)
        except queue.Empty:
            continue  # 没有图像输入，继续等待


def main():
    rospy.init_node('find_signal')

    signal_center_pub = rospy.Publisher('/signal_center', Point32, queue_size=10)
    signal_text_pub = rospy.Publisher('/signal_text', String, queue_size=10)
    rospy.Subscriber("/ucar_camera/image_raw", Image, image_callback, queue_size=1)

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

        while not rospy.is_shutdown():
            if ros_image is None:
                rospy.sleep(0.01)
                continue

            frame = ros_image.copy()
            frame = img_proc.letter_box(frame, DET_INPUT_SHAPE)[0]
            img_proc.queue_img(frame, input_queue)

            try:
                cur_t = time.perf_counter()
                elapsed += cur_t - pre_t
                frame_count += 1
                pre_t = cur_t

                if frame_count >= 10:
                    fps = frame_count / elapsed if elapsed > 0 else 0.0
                    img_proc.update_fps(fps)
                    elapsed = 0
                    frame_count = 0

                det_output = det_output_queue.get(timeout=0.15)  # 获取检测结果
                det_output = np.array(det_output).astype(np.int32)
                center = get_center_point(det_output)
                logging.info(f"Center point: {center}")

                # 发布 center
                pt = Point32()
                pt.x = float(center[0])
                pt.y = float(center[1])
                pt.z = 0.0
                signal_center_pub.publish(pt)

                rec_output = rec_output_queue.get(timeout=0.1)  # 获取识别结果
                logging.info(f"Recognition result: {rec_output}")

                # 发布 rec_output
                if rec_output and len(rec_output) > 0:
                    text, conf = rec_output[0]
                    msg = String()
                    msg.data = json.dumps({"text": text, "confidence": float(conf)})
                    signal_text_pub.publish(msg)

            except queue.Empty:
                pass

    except rospy.ROSInterruptException:
        pass

    finally:
        for det_model in det_models:
            det_model.release()
        for rec_model in rec_models:
            rec_model.release()


if __name__ == "__main__":
    main()
