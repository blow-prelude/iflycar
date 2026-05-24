import argparse
import os
import time

import cv2
import numpy as np
import operators
from db_postprocess import DBPostProcess, DetPostProcess

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录
parent_dir = os.path.dirname(script_dir)

# 模型文件路径（在上一级的models/下）
DET_MODEL = os.path.join(parent_dir, "models", "ch_PP-OCRv4_det_infer.rknn")
REC_MODEL = os.path.join(parent_dir, "models", "ch_PP-OCRv4_rec_infer.rknn")

# 输入图像路径
IMG_PATH = os.path.join(script_dir, "test.jpg")

# 输出目录（在上一级的output/下）
OUTPUT_DIR = os.path.join(parent_dir, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DET_INPUT_SHAPE = [480, 480]  # h,w

RKNN_PRE_PROCESS_CONFIG = [
    {"DetResizeForTest": {"image_shape": DET_INPUT_SHAPE}},
    {
        "NormalizeImage": {
            "std": [1.0, 1.0, 1.0],
            "mean": [0.0, 0.0, 0.0],
            "scale": "1.",
            "order": "hwc",
        }
    },
]
POSTPROCESS_CONFIG = {
    "DBPostProcess": {
        "thresh": 0.3,
        "box_thresh": 0.6,
        "max_candidates": 1000,
        "unclip_ratio": 1.5,
        "use_dilation": False,
        "score_mode": "fast",
    }
}


class TextDetector:
    def __init__(self, args) -> None:
        self.model, self.framework = setup_model(args)
        self.preprocess_funct = []
        PRE_PROCESS_CONFIG = RKNN_PRE_PROCESS_CONFIG

        for item in PRE_PROCESS_CONFIG:
            for key in item:
                pclass = getattr(operators, key)
                p = pclass(**item[key])
                self.preprocess_funct.append(p)

        self.db_postprocess = DBPostProcess(**POSTPROCESS_CONFIG["DBPostProcess"])
        self.det_postprocess = DetPostProcess()

    def preprocess(self, img):
        for p in self.preprocess_funct:
            img = p(img)
        return img

    def run(self, img):
        model_input = self.preprocess({"image": img})
        output = self.model.run([model_input["image"]])

        preds = {"maps": output[0].astype(np.float32)}
        result = self.db_postprocess(preds, model_input["shape"])

        output = self.det_postprocess.filter_tag_det_res(result[0]["points"], img.shape)
        return output


def setup_model(args):
    model_path = args.model_path
    if model_path.endswith(".rknn"):
        platform = "rknn"
        from rknn_executor import RKNN_model_container

        model = RKNN_model_container(model_path, args.target, args.device_id)
    else:
        assert False, "{} is not rknn/onnx model".format(model_path)
    print("Model-{} is {} model, starting val".format(model_path, platform))
    return model, platform


def init_args():
    parser = argparse.ArgumentParser(description="PPOCR-Det Python Demo")
    # basic params
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="model path, could be .onnx or .rknn file",
    )
    parser.add_argument(
        "--target", type=str, default="rk3566", help="target RKNPU platform"
    )
    parser.add_argument("--device_id", type=str, default=None, help="device id")
    return parser


if __name__ == "__main__":
    # Init model
    parser = init_args()
    args = parser.parse_args()
    det_model = TextDetector(args)

    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (DET_INPUT_SHAPE[1], DET_INPUT_SHAPE[0]))

    time1 = time.perf_counter()
    outout = det_model.run(img)
    time2 = time.perf_counter()
    print(f"Inference time: {time2 - time1:.4f} s")

    for box in outout:
        box = np.array(box).astype(np.int32)
        cv2.polylines(img, [box], True, (0, 255, 0), 2)

    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    print(outout.tolist())
