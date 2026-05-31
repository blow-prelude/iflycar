#!/home/ucar/venv3.9/bin/python3
import os
import queue
import threading
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from rknn_det import TextDetector
from rknn_rec import TextRecognizer
from rknnlite.api import RKNNLite
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int32

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
        pts = np.array(det_output).astype(np.float64)

        # 处理嵌套结构：如果是3D数组 (N, M, 2)，提取第一个框
        if pts.ndim == 3 and pts.shape[0] == 1:
            pts = pts[0]  # (1, 4, 2) -> (4, 2)
        # 反向letterbox: 去padding, 去缩放
        pts[:, 0] = (pts[:, 0] - dw) / lb_ratio
        pts[:, 1] = (pts[:, 1] - dh) / lb_ratio
        # 缩放到display尺寸
        pts[:, 0] *= disp_shape[1] / orig_shape[1]
        pts[:, 1] *= disp_shape[0] / orig_shape[0]
        return pts.astype(np.int32).reshape(det_output.shape)


def get_center_point(box):
    x_mid = int((box[:, 0].min() + box[:, 0].max()) / 2)
    y_mid = int((box[:, 1].min() + box[:, 1].max()) / 2)
    return (x_mid, y_mid)


def get_biggest_box(boxes):
    # 转换为数组并检查是否为空
    boxes = np.asarray(boxes)

    # 如果数组为空或维度不足（无检测结果），直接返回 None
    if boxes.size == 0 or boxes.ndim < 2:
        return None

    boxes = boxes.astype(np.int32)

    # 确保至少是3D数组：如果是2D数组(单个框)，添加一个维度
    if boxes.ndim == 2:
        boxes = boxes[np.newaxis, :]  # (4, 2) -> (1, 4, 2)

    # 沿着轴 1 一次性求出所有 box 的最小/最大坐标
    # mins 和 maxs 的形状均为 (N, 2)，对应每行 [x_min/x_max, y_min/y_max]
    mins = boxes.min(axis=1)
    maxs = boxes.max(axis=1)

    # 3. 向量化计算所有面积
    areas = (maxs[:, 0] - mins[:, 0]) * (maxs[:, 1] - mins[:, 1])

    # 4. 获取最大面积的索引并取出对应的 box
    max_idx = np.argmax(areas) if len(areas) > 0 else None
    max_box = boxes[max_idx] if max_idx is not None else None
    ordered_box = order_corners(max_box) if max_box is not None else None
    return ordered_box


def order_corners(det_output):
    """返回检测框4个角点，顺序: 左上、右上、右下、左下"""
    tl = det_output[np.argmin(det_output[:, 0] + det_output[:, 1])]
    br = det_output[np.argmax(det_output[:, 0] + det_output[:, 1])]
    tr = det_output[np.argmin(det_output[:, 0] - det_output[:, 1])]
    bl = det_output[np.argmax(det_output[:, 0] - det_output[:, 1])]
    return np.array([tl, tr, br, bl])


def crop_roi(img, roi):
    # roi按照左上，右上，右下，左下的顺序排列

    if roi is not None:
        x_min = roi[:, 0].min()
        x_max = roi[:, 0].max()
        y_min = roi[:, 1].min()
        y_max = roi[:, 1].max()
        cropped = img[y_min:y_max, x_min:x_max]
        # logging.debug(
        #     f"Largest box: ({x_min}, {y_min}, {x_max}, {y_max}), size: {x_max - x_min}x{y_max - y_min}"
        # )
    else:
        cropped = None
    return cropped


def classfy(text):
    if any(kw in text for kw in ["食品", "食"]):
        return 0
    if any(kw in text for kw in ["日用品", "日", "用品"]):
        return 1
    if any(kw in text for kw in ["电子产品", "电子", "电", "生产"]):
        return 2
    return -1


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
                det_output_queue.put([biggest_box])

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

        signal_pub = rospy.Publisher(
            "/signal_detection", Float32MultiArray, queue_size=10
        )
        signal_class_pub = rospy.Publisher("/signal_class", Int32, queue_size=10)
        rospy.Subscriber("/ucar_camera/image_raw", Image, image_callback, queue_size=1)

        det_model0 = TextDetector(target="rk3588", device_id=RKNNLite.NPU_CORE_0)
        rec_model0 = TextRecognizer(target="rk3588", device_id=RKNNLite.NPU_CORE_0)
        det_model1 = TextDetector(target="rk3588", device_id=RKNNLite.NPU_CORE_1)
        rec_model1 = TextRecognizer(target="rk3588", device_id=RKNNLite.NPU_CORE_1)
        det_model2 = TextDetector(target="rk3588", device_id=RKNNLite.NPU_CORE_2)
        rec_model2 = TextRecognizer(target="rk3588", device_id=RKNNLite.NPU_CORE_2)
        rospy.loginfo("Models initialized successfully")

        img_proc = img_processor()

        det_models = [det_model0, det_model1, det_model2]
        rec_models = [rec_model0, rec_model1, rec_model2]
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

                img_proc.draw_fps(canvas)
                cv2.imshow("Detection Result", canvas)
                cv2.waitKey(1)
                # # 发布 center
                # pt = Point32()
                # pt.x = float(center[0])
                # pt.y = float(center[1])
                # pt.z = 0.0
                # signal_center_pub.publish(pt)

                # # 发布 box_x_l, box_x_r
                # box_msg = Point32()
                # box_msg.x = float(box_x_l)
                # box_msg.y = float(box_x_r)
                # box_msg.z = 0.0
                # signal_box_pub.publish(box_msg)

                data = [
                    float(center[0]),
                    float(center[1]),
                    float(box_x_l),
                    float(box_x_r),
                ]
                msg = Float32MultiArray(data=data)
                signal_pub.publish(msg)

                rec_output = rec_output_queue.get_nowait()  # 获取识别结果
                rospy.loginfo(f"Recognition result: {rec_output}")

                # 发布 rec_output
                if rec_output and len(rec_output) > 0:
                    text = rec_output[0][0]  # 识别出的字符串，如 "电子产品生产车间"
                    class_id = classfy(text)  # 调用分类得到 2
                    rospy.loginfo(f"text: {text}, class_id: {class_id}")
                    msg = Int32()
                    if class_id != -1:
                        msg.data = class_id
                        signal_class_pub.publish(msg)

            except queue.Empty:
                img_proc.draw_fps(canvas)
                cv2.imshow("Detection Results", canvas)
                cv2.waitKey(1)

    except rospy.ROSInterruptException:
        pass

    finally:
        rospy.loginfo("Shutting down, releasing resources...")
        for det_model in det_models:
            det_model.release()
        for rec_model in rec_models:
            rec_model.release()


if __name__ == "__main__":
    main()
