#!/home/ucar/venv3.9/bin/python3
import os
import queue
import threading
import time

import cv2
import numpy as np
import rospy
from camera_capture import CameraCapture
from coco_utils import COCO_test_helper
from cv_bridge import CvBridge, CvBridgeError
from rknn_executor import RKNN_model_container
from rknnlite.api import RKNNLite
from sensor_msgs.msg import Image
from std_msgs.msg import String


OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (640, 480)  # (width, height), such as (1280, 736)

CLASSES = ("stop", "straight", "right", "left")

coco_id_list = [
    1,
    2,
    3,
    4,
]


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold."""
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def dfl(position):
    # Distribution Focal Loss (DFL) - NumPy implementation
    x = position.astype(np.float32)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)

    # Softmax along the mc dimension (axis=2)
    exp_y = np.exp(y - np.max(y, axis=2, keepdims=True))
    y = exp_y / np.sum(exp_y, axis=2, keepdims=True)

    # Weighted sum using cumulative indices
    acc_matrix = np.arange(mc).astype(np.float32).reshape(1, 1, mc, 1, 1)
    y = np.sum(y * acc_matrix, axis=2)
    return y


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(
        1, 2, 1, 1
    )

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def post_process(input_data):
    num_branches = 3
    pair_per_branch = len(input_data) // num_branches

    # 处理每个检测分支: box回归 + 类别置信度
    box_parts, conf_parts = [], []
    for i in range(num_branches):
        box_parts.append(box_process(input_data[pair_per_branch * i]))
        conf_parts.append(input_data[pair_per_branch * i + 1])

    # (1, C, H, W) -> (H*W, C)
    def flatten_hw(x):
        return x.transpose(0, 2, 3, 1).reshape(-1, x.shape[1])

    boxes = np.concatenate([flatten_hw(b) for b in box_parts])
    classes_conf = np.concatenate([flatten_hw(c) for c in conf_parts])

    # box_confidence 全为1，实际置信度来自 class_max_score * 1
    box_confidences = np.ones((boxes.shape[0], 1), dtype=np.float32)

    # 按阈值过滤
    boxes, classes, scores = filter_boxes(boxes, box_confidences, classes_conf)

    if boxes is None or len(boxes) == 0:
        return None, None, None

    # 按类别做NMS
    nboxes, nclasses, nscores = [], [], []
    for cls_id in np.unique(classes):
        mask = classes == cls_id
        cls_boxes = boxes[mask]
        cls_scores = scores[mask]
        keep = nms_boxes(cls_boxes, cls_scores)

        if len(keep) > 0:
            idx = np.where(mask)[0][keep]
            nboxes.append(boxes[idx])
            nclasses.append(classes[idx])
            nscores.append(scores[idx])

    if not nboxes:
        return None, None, None

    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print(
            "%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score)
        )
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(
            image,
            "{0} {1:.2f}".format(CLASSES[cl], score),
            (top, left - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )


def setup_model(model_path, target="rk3588", device_id=RKNNLite.NPU_CORE_0_1):
    if model_path.endswith(".rknn"):
        platform = "rknn"
        model = RKNN_model_container(model_path, target, device_id)
    else:
        raise ValueError(f"no model in {model_path}")
    print("Model-{} is {} model, starting val".format(model_path, platform))
    return model, platform


def inference_worker(
    worker_id, model, input_queue, output_queue, co_helper, stop_event
):
    """推理工作线程函数

    Args:
        worker_id: 工作线程ID
        model: 绑定的RKNN模型
        input_queue: 输入图像队列
        output_queue: 输出结果队列
        co_helper: COCO辅助工具
        stop_event: 停止事件
    """
    rospy.loginfo(f"Worker-{worker_id} started on NPU core {worker_id}")

    while not stop_event.is_set():
        try:
            # 从队列获取图像，超时0.1秒
            img_rgb = input_queue.get(timeout=0.1)

            # 执行推理
            time1 = time.perf_counter()
            outputs = model.run([img_rgb])
            time2 = time.perf_counter()
            # rospy.logdebug(f"Worker-{worker_id} inference time: {time2 - time1:.4f} s")

            # 后处理
            boxes, classes, scores = post_process(outputs)
            # rospy.logdebug(
            #     f"Worker-{worker_id} post-process time: {time.perf_counter() - time2:.4f} s"
            # )

            # 绘制结果
            canvas = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
            if boxes is not None:
                draw(canvas, co_helper.get_real_box(boxes), scores, classes)

            # 放入结果队列
            result = {
                "canvas": canvas,
                "inference_time": time2 - time1,
                "worker_id": worker_id,
                "classes": classes,
                "scores": scores,
            }
            output_queue.put(result)

            input_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            rospy.logerr(f"Worker-{worker_id} error: {e}")
            continue

    rospy.loginfo(f"Worker-{worker_id} stopped")


def img_check(path):
    img_type = [".jpg", ".jpeg", ".png", ".bmp"]
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False


class ImgProcessor:
    def __init__(self, co_helper):
        self.co_helper = co_helper
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

    def preprocess(self, img_src):
        pad_color = (0, 0, 0)
        img_rgb = self.co_helper.letter_box(
            im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=pad_color
        )
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        return img_rgb

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


def main():
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(FILE_DIR, "..", "models", "bestfp.rknn")

    # 创建队列
    input_queue = queue.Queue(maxsize=10)  # 图像队列
    output_queue = queue.Queue(maxsize=10)  # 结果队列
    stop_event = threading.Event()

    # 用于FPS统计
    display_frame_count = 0  # 用于FPS显示更新
    fps_start_time = time.time()

    model0 = model1 = model2 = None
    workers = []

    try:
        # 跳过 ROS 的 fileConfig（与 Python 3.9 不兼容）
        import logging.config
        _orig_fileConfig = logging.config.fileConfig
        logging.config.fileConfig = lambda *a, **kw: None
        rospy.init_node("judge_light", anonymous=True)
        logging.config.fileConfig = _orig_fileConfig
        direction_topic = rospy.get_param("~direction_topic", "/vision_line_direction")
        direction_pub = rospy.Publisher(direction_topic, String, queue_size=10)
        rospy.loginfo(f"Publishing direction to: {direction_topic}")

        # 初始化3个模型，分别绑定到3个NPU核心
        rospy.loginfo("Initializing 3 models on NPU cores 0, 1, 2...")
        model0, _ = setup_model(MODEL_PATH, device_id=RKNNLite.NPU_CORE_0)
        model1, _ = setup_model(MODEL_PATH, device_id=RKNNLite.NPU_CORE_1)
        model2, _ = setup_model(MODEL_PATH, device_id=RKNNLite.NPU_CORE_2)

        # 初始化摄像头（ROS 话题）
        image_topic = rospy.get_param("~image_topic", "ucar_camera/image_raw")
        bridge = CvBridge()
        latest_frame = None
        frame_lock = threading.Lock()

        def image_callback(msg):
            nonlocal latest_frame
            try:
                frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                with frame_lock:
                    latest_frame = frame
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge conversion failed: {e}")

        rospy.Subscriber(image_topic, Image, image_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo(f"Subscribed image topic: {image_topic}")

        co_helper = COCO_test_helper(enable_letter_box=True)
        img_processor = ImgProcessor(co_helper)

        # 创建并启动3个工作线程
        models = [model0, model1, model2]
        workers = []
        for i in range(3):
            worker = threading.Thread(
                target=inference_worker,
                args=(i, models[i], input_queue, output_queue, co_helper, stop_event),
                daemon=True,
            )
            worker.start()
            workers.append(worker)
            rospy.loginfo(f"Started worker-{i}")

        rospy.loginfo("Starting main loop...")
        loop_rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            # 主线程：从 ROS 话题获取图像
            with frame_lock:
                img_src = latest_frame.copy() if latest_frame is not None else None
            if img_src is None:
                loop_rate.sleep()
                continue

            # 预处理图像
            img_rgb = img_processor.preprocess(img_src)

            # 将预处理后的图像放入输入队列
            img_processor.queue_img(img_rgb, input_queue)

            # 从输出队列获取结果并显示
            try:
                result = output_queue.get(timeout=0.1)
                canvas = result["canvas"]
                inference_time = result["inference_time"]
                worker_id = result["worker_id"]
                classes = result["classes"]
                scores = result["scores"]
                best_score = (
                    scores.max() if scores is not None and len(scores) > 0 else 0
                )
                best_class = (
                    classes[scores.argmax()]
                    if classes is not None and scores is not None and len(classes) > 0
                    else None
                )

                rospy.loginfo(
                    f"worker-{worker_id}: class: {best_class}, scores: {best_score} ,inference time: {inference_time:.4f} s"
                )

                # 发布检测到的方向到 /vision_line_direction 话题
                if best_class is not None and best_score >= OBJ_THRESH:
                    direction_name = CLASSES[best_class]
                    direction_pub.publish(String(direction_name))
                    rospy.loginfo(f"Detected direction: {direction_name}, shutting down...")
                    # 等待消息被消费，避免订阅者未收到就退出
                    time.sleep(1.0)
                    break

                # 统计FPS
                current_time = time.perf_counter()
                display_frame_count += 1
                # 每10帧更新一次FPS显示
                if display_frame_count >= 10:
                    elapsed = current_time - fps_start_time
                    fps = display_frame_count / elapsed if elapsed > 0 else 0
                    rospy.loginfo(f"Current FPS: {fps:.2f}")
                    img_processor.update_fps(fps)
                    fps_start_time = current_time
                    display_frame_count = 0

                img_processor.draw_fps(canvas)
                cv2.imshow("predict", canvas)
                cv2.waitKey(1)

            except queue.Empty:
                # 队列为空时继续显示原始图像，使用上次的FPS
                img_processor.draw_fps(img_src)
                cv2.imshow("img_src", img_src)
                cv2.waitKey(1)
            except Exception as e:
                rospy.logerr(f"Error displaying result: {e}")

    except KeyboardInterrupt:
        rospy.loginfo("Interrupt by user, exiting...")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        # 停止所有工作线程
        stop_event.set()
        for worker in workers:
            worker.join(timeout=2.0)

        # 释放资源
        cv2.destroyAllWindows()
        if model0: model0.release()
        if model1: model1.release()
        if model2: model2.release()


def main_test():
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(FILE_DIR, "..", "models", "bestfp.rknn")
    img_path = os.path.join(FILE_DIR, "..", "pictures", "006_0018.jpg")

    try:
        # init model
        model, _ = setup_model(MODEL_PATH, device_id=RKNNLite.NPU_CORE_0)
    except Exception as e:
        rospy.logerr(f"occur err when setup model: {e}")

    try:
        co_helper = COCO_test_helper(enable_letter_box=True)

        pad_color = (0, 0, 0)

        img_src = cv2.imread(img_path)
        # 将图像等比例缩放并填充到制指定尺寸
        img = co_helper.letter_box(
            im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=pad_color
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 将红灯stop的状态也发布
        time1 = time.perf_counter()
        outputs = model.run([img])
        time2 = time.perf_counter()
        rospy.loginfo(f"inference take {time2 - time1:.6f}s")
        boxes, classes, scores = post_process(outputs)
        rospy.loginfo(f"post process takes {time.perf_counter() - time2:.6f}s")

        canvas = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
        if boxes is not None:
            draw(canvas, co_helper.get_real_box(boxes), scores, classes)

        cv2.imshow("predict", canvas)
        cv2.waitKey(0)
    except KeyboardInterrupt:
        rospy.loginfo("Interrupt by user,exiting...")
    except Exception as e:
        rospy.logerr(f"occur err when inference: {e}")
    finally:
        cv2.destroyAllWindows()
        model.release()


if __name__ == "__main__":
    main()
