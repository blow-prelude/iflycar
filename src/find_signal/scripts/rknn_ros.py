#!/home/ucar/venv3.9/bin/python3
import os
import queue
import threading
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point32
from rknn_det import TextDetector
from rknn_rec import TextRecognizer
from rknnlite.api import RKNNLite
from sensor_msgs.msg import Image
from std_msgs.msg import String

# Configure logging

IMG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.jpg")
DET_INPUT_SHAPE = [480, 480]
REC_INPUT_SHAPE = [48, 320]
DISP_SHAPE = [480, 640]

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

    def transform_to_display(
        self, det_output, lb_ratio, lb_padding, orig_shape, disp_shape=DISP_SHAPE
    ):
        """将letterboxed坐标(480x480)转换到display坐标(640x480)"""
        dw, dh = lb_padding
        pts = det_output.reshape(-1, 2).astype(np.float64)
        # 反向letterbox: 去padding, 去缩放
        pts[:, 0] = (pts[:, 0] - dw) / lb_ratio
        pts[:, 1] = (pts[:, 1] - dh) / lb_ratio
        # 缩放到display尺寸
        pts[:, 0] *= disp_shape[1] / orig_shape[1]
        pts[:, 1] *= disp_shape[0] / orig_shape[0]
        return pts.astype(np.int32).reshape(det_output.shape)


def get_center_point(box):
    box = np.array(box).reshape(-1, 2)
    x_mid = int((box[:, 0].min() + box[:, 0].max()) / 2)
    y_mid = int((box[:, 1].min() + box[:, 1].max()) / 2)
    return (x_mid, y_mid)


def get_biggest_box(boxes):
    # 转换为 numpy 数组
    boxes = np.asarray(boxes)

    # 如果没有检测到任何框，或者形状不符合预期，直接返回 None
    if boxes.size == 0 or boxes.ndim < 2:
        return None

    # 如果输入是二维数组 (N, M)，但期望是 (N, 顶点数, 2)，可以尝试 reshape
    # 这里假设每个框由 8 个值组成 (4 个点 × 2 坐标)
    if boxes.ndim == 2:
        # 如果最后一个维度不是 2，尝试重塑
        if boxes.shape[-1] != 2:
            boxes = boxes.reshape(boxes.shape[0], -1, 2)
    # 如果是一维的单个框，也重塑为 (1, -1, 2)
    elif boxes.ndim == 1:
        boxes = boxes.reshape(1, -1, 2)

    # 现在 boxes 应该是 (N, 顶点数, 2)，可以安全使用 axis=1
    mins = boxes.min(axis=1).astype(int)
    maxs = boxes.max(axis=1).astype(int)

    areas = (maxs[:, 0] - mins[:, 0]) * (maxs[:, 1] - mins[:, 1])
    max_idx = np.argmax(areas) if len(areas) > 0 else None
    max_box = boxes[max_idx] if max_idx is not None else None

    if max_box is not None:
        return order_corners(max_box)
    return None


def order_corners(det_output):
    """返回检测框4个角点，顺序: 左上、右上、右下、左下"""
    pts = det_output.reshape(-1, 2).astype(np.float64)
    center = pts.mean(axis=0)
    tl = pts[(pts[:, 0] < center[0]) & (pts[:, 1] < center[1])]
    tr = pts[(pts[:, 0] >= center[0]) & (pts[:, 1] < center[1])]
    br = pts[(pts[:, 0] >= center[0]) & (pts[:, 1] >= center[1])]
    bl = pts[(pts[:, 0] < center[0]) & (pts[:, 1] >= center[1])]
    return np.array([tl[0], tr[0], br[0], bl[0]], dtype=np.int32)


def crop_roi(img, roi):
    # roi按照左上，右上，右下，左下的顺序排列

    if roi is not None:
        x_min = int(roi[:, 0].min())
        x_max = int(roi[:, 0].max())
        y_min = int(roi[:, 1].min())
        y_max = int(roi[:, 1].max())
        cropped = img[y_min:y_max, x_min:x_max]
        # logging.debug(
        #     f"Largest box: ({x_min}, {y_min}, {x_max}, {y_max}), size: {x_max - x_min}x{y_max - y_min}"
        # )
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
            # logging.debug(f"det inference time: {time.perf_counter() - time1:.4f} s")
            biggest_box = get_biggest_box(det_output)
            if biggest_box is not None:
                det_output_queue.put([biggest_box.astype(np.int32)])

            cropped = crop_roi(img, biggest_box)
            if cropped is not None:
                cropped = cv2.resize(cropped, (REC_INPUT_SHAPE[1], REC_INPUT_SHAPE[0]))
                time1 = time.perf_counter()
                rec_output = rec_model.run(cropped)
                # logging.debug(
                #     f"rec inference time: {time.perf_counter() - time1:.4f} s"
                # )
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

        import logging.config

        _orig_fileConfig = logging.config.fileConfig
        logging.config.fileConfig = lambda *a, **kw: None
        rospy.init_node("find_signal", anonymous=True)
        logging.config.fileConfig = _orig_fileConfig

        signal_center_pub = rospy.Publisher("/signal_center", Point32, queue_size=10)
        signal_text_pub = rospy.Publisher("/signal_text", String, queue_size=10)
        signal_box_pub = rospy.Publisher("/signal_box", Point32, queue_size=10)
        rospy.Subscriber("/ucar_camera/image_raw", Image, image_callback, queue_size=1)

        det_model0 = TextDetector(target="rk3588", device_id=RKNNLite.NPU_CORE_0_1)
        rec_model0 = TextRecognizer(target="rk3588", device_id=RKNNLite.NPU_CORE_0_1)
        det_model1 = TextDetector(target="rk3588", device_id=RKNNLite.NPU_CORE_2)
        rec_model1 = TextRecognizer(target="rk3588", device_id=RKNNLite.NPU_CORE_2)
        # det_model2 = TextDetector(target="rk3588", device_id=RKNNLite.NPU_CORE_2)
        # rec_model2 = TextRecognizer(target="rk3588", device_id=RKNNLite.NPU_CORE_2)
        rospy.loginfo("Models initialized successfully")

        img_proc = img_processor()

        det_models = [det_model0, det_model1]
        rec_models = [rec_model0, rec_model1]
        workers = []

    except Exception as e:
        rospy.logerr(f"Error init: {e}")
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
            orig_shape = frame.shape[:2]
            lb_frame, lb_ratio, lb_padding = img_proc.letter_box(frame, DET_INPUT_SHAPE)
            canvas = cv2.resize(frame, (DISP_SHAPE[1], DISP_SHAPE[0]))

            # cv2.imshow("before detection", frame)
            img_proc.queue_img(lb_frame, input_queue)
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

                det_output = det_output_queue.get_nowait()

                det_output = np.array(det_output)

                # logging.debug(f"Detection box: {det_output}")

                det_output = np.array(det_output)
                det_output = img_proc.transform_to_display(
                    det_output, lb_ratio, lb_padding, orig_shape
                )
                box_x_l = det_output[:, 0].min()
                box_x_r = det_output[:, 0].max()
                rospy.loginfo(f"box_x_l: {box_x_l}, box_x_r: {box_x_r}")
                center = get_center_point(det_output)
                # logging.debug(f"Drawing box: {det_output}")
                rospy.loginfo(f"Center point: {center}")
                cv2.polylines(canvas, [det_output], True, (0, 255, 0), 2)

                # 发布 center
                pt = Point32()
                pt.x = float(center[0])
                pt.y = float(center[1])
                pt.z = 0.0
                signal_center_pub.publish(pt)

                # 发布 box_x_l, box_x_r
                box_msg = Point32()
                box_msg.x = float(box_x_l)
                box_msg.y = float(box_x_r)
                box_msg.z = 0.0
                signal_box_pub.publish(box_msg)

                rec_output = rec_output_queue.get(timeout=0.1)  # 获取识别结果
                rospy.loginfo(f"Recognition result: {rec_output}")

                # 发布 rec_output
                if rec_output and len(rec_output) > 0:
                    text = rec_output[0][0]
                    msg = String()
                    msg.data = text
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
