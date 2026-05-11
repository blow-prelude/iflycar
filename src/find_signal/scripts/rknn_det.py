import os
import time

import cv2
import numpy as np
import operators
from db_postprocess import DBPostProcess, DetPostProcess
from rknnlite.api import RKNNLite

# add path
# realpath = os.path.abspath(__file__)
# _sep = os.path.sep
# realpath = realpath.split(_sep)
# sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(FILE_DIR, "..", "models")

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
    def __init__(self, target="rk3588", device_id=RKNNLite.NPU_CORE_0_1) -> None:
        self.model_path = os.path.join(MODEL_DIR, "ppocrv4_det.rknn")
        self.target = target
        self.device_id = device_id

        self.model, self.framework = self.setup_model()
        # 构建预处理链
        self.preprocess_funct = []
        PRE_PROCESS_CONFIG = RKNN_PRE_PROCESS_CONFIG
        for item in PRE_PROCESS_CONFIG:
            for key in item:
                pclass = getattr(operators, key)
                p = pclass(**item[key])
                self.preprocess_funct.append(p)
        # 初始化后处理器
        self.db_postprocess = DBPostProcess(**POSTPROCESS_CONFIG["DBPostProcess"])
        self.det_postprocess = DetPostProcess()

    def setup_model(self):
        platform = "rknn"
        from rknn_executor import RKNN_model_container

        model = RKNN_model_container(self.model_path, self.target, self.device_id)
        return model, platform

    def preprocess(self, img):
        for p in self.preprocess_funct:
            img = p(img)
        return img

    def run(self, img):
        # time1 = time.perf_counter()
        model_input = self.preprocess({"image": img})
        # print(f"preprocess time: {time.perf_counter() - time1:.4f} s")

        # time1 = time.perf_counter()
        output = self.model.run([model_input["image"]])
        # print(f"inference time: {time.perf_counter() - time1:.4f} s")

        preds = {"maps": output[0].astype(np.float32)}

        # time1 = time.perf_counter()
        result = self.db_postprocess(preds, model_input["shape"])
        output = self.det_postprocess.filter_tag_det_res(result[0]["points"], img.shape)
        # print(f"postprocess time: {time.perf_counter() - time1:.4f} s")

        return output

    def release(self):
        self.model.release()


if __name__ == "__main__":
    try:
        det_model = TextDetector(target="rk3588", device_id=RKNNLite.NPU_CORE_0_1)

        # Set inputs
        img_path = os.path.join(FILE_DIR, "test.jpg")
        img = cv2.imread(img_path)
        img = cv2.resize(img, (DET_INPUT_SHAPE[1], DET_INPUT_SHAPE[0]))
        # cv2.imshow("img", img)
        # cv2.waitKey(0)

        # Inference
        time1 = time.perf_counter()
        output = det_model.run(img)
        print(f"total inference time: {time.perf_counter() - time1:.4f} s")

        # Post Process
        for box in output:
            box = np.array(box).astype(np.int32)
            cv2.polylines(img, [box], True, (0, 255, 0), 2)
        cv2.imshow("img", img)
        cv2.waitKey(0)

        print(output.tolist())

    finally:
        det_model.release()
        cv2.destroyAllWindows()
