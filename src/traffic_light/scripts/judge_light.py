import logging
import os
import time

import cv2
import numpy as np
from camera_capture import CameraCapture
from coco_utils import COCO_test_helper
from rknn_executor import RKNN_model_container
from rknnlite.api import RKNNLite

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (640, 480)  # (width, height), such as (1280, 736)

CLASSES = ("stop", "straight", "left", "right")

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
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(
            np.ones_like(
                input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32
            )
        )

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


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


def img_check(path):
    img_type = [".jpg", ".jpeg", ".png", ".bmp"]
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False


def main():
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(FILE_DIR, "..", "models", "best.rknn")

    try:
        # init model
        model, _ = setup_model(MODEL_PATH, device_id=RKNNLite.NPU_CORE_0_1)
        cap = CameraCapture(0, 640, 480)
    except Exception as e:
        logging.error(f"occur err when init: {e}")

    try:
        co_helper = COCO_test_helper(enable_letter_box=True)

        pad_color = (0, 0, 0)
        while True:
            img_src = cap.get_picture()
            img_src = cap.correct_img(img_src)
            img_src = cv2.flip(img_src, 1)
            # 将图像等比例缩放并填充到制指定尺寸
            img = co_helper.letter_box(
                im=img_src.copy(),
                new_shape=(IMG_SIZE[1], IMG_SIZE[0]),
                pad_color=pad_color,
            )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            time1 = time.perf_counter()
            outputs = model.run([img])
            time2 = time.perf_counter()
            logging.info(f"inference take {time2 - time1:.6f}s")
            boxes, classes, scores = post_process(outputs)
            logging.info(f"post process takes {time.perf_counter() - time2:.6f}s")

            canvas = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
            if boxes is not None:
                draw(canvas, co_helper.get_real_box(boxes), scores, classes)

            cv2.imshow("predict", canvas)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        logging.info("Interrupt by user,exiting...")
    except Exception as e:
        logging.error(f"occur err when inference: {e}")
    finally:
        cv2.destroyAllWindows()
        cap.close()
        model.release()


def main_test():
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(FILE_DIR, "..", "models", "best.rknn")
    img_path = os.path.join(FILE_DIR, "..", "pictures", "006_0018.jpg")

    try:
        # init model
        model, _ = setup_model(MODEL_PATH, device_id=RKNNLite.NPU_CORE_0)
    except Exception as e:
        logging.error(f"occur err when setup model: {e}")

    try:
        co_helper = COCO_test_helper(enable_letter_box=True)

        pad_color = (0, 0, 0)

        img_src = cv2.imread(img_path)
        # 将图像等比例缩放并填充到制指定尺寸
        img = co_helper.letter_box(
            im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=pad_color
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        time1 = time.perf_counter()
        outputs = model.run([img])
        time2 = time.perf_counter()
        logging.info(f"inference take {time2 - time1:.6f}s")
        boxes, classes, scores = post_process(outputs)
        logging.info(f"post process takes {time.perf_counter() - time2:.6f}s")

        canvas = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
        if boxes is not None:
            draw(canvas, co_helper.get_real_box(boxes), scores, classes)

        cv2.imshow("predict", canvas)
        cv2.waitKey(0)
    except KeyboardInterrupt:
        logging.info("Interrupt by user,exiting...")
    except Exception as e:
        logging.error(f"occur err when inference: {e}")
    finally:
        cv2.destroyAllWindows()
        model.release()


if __name__ == "__main__":
    # main()
    main_test()
