# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

import cv2
import numpy as np
import operators
from rec_postprocess import CTCLabelDecode
from rknnlite.api import RKNNLite

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录
parent_dir = os.path.dirname(script_dir)

# 输入图像路径
IMG_PATH = os.path.join(script_dir, "test11.png")
MODEL_DIR = os.path.join(script_dir, "..", "models")

REC_INPUT_SHAPE = [48, 320]  # h,w
CHARACTER_DICT_PATH = os.path.join(parent_dir, "models", "ppocr_keys_v1.txt")

PRE_PROCESS_CONFIG = [
    {
        "NormalizeImage": {
            "std": [1, 1, 1],
            "mean": [0, 0, 0],
            "scale": "1./255.",
            "order": "hwc",
        }
    }
]

POSTPROCESS_CONFIG = {
    "CTCLabelDecode": {
        "character_dict_path": CHARACTER_DICT_PATH,
        "use_space_char": True,
    }
}


class TextRecognizer:
    def __init__(self, target="rk3588", device_id=RKNNLite.NPU_CORE_0_1) -> None:
        self.model_path = os.path.join(MODEL_DIR, "ppocrv4_rec.rknn")
        self.target = target
        self.device_id = device_id

        self.model, self.framework = self.setup_model()
        # 构建预处理链
        self.preprocess_funct = []
        for item in PRE_PROCESS_CONFIG:
            for key in item:
                pclass = getattr(operators, key)
                p = pclass(**item[key])
                self.preprocess_funct.append(p)
        # 初始化后处理器
        self.ctc_postprocess = CTCLabelDecode(**POSTPROCESS_CONFIG["CTCLabelDecode"])

    def preprocess(self, img):
        for p in self.preprocess_funct:
            img = p(img)
        return img

    def run(self, img):
        model_input = self.preprocess({"image": img})
        output = self.model.run([model_input["image"]])
        preds = output[0].astype(np.float32)

        # RKNN模型输出可能是 [1, seq_len, num_classes, 1]
        # 需要转换为 [batch, seq_len, num_classes]
        if len(preds.shape) == 4:
            # 去掉最后一维: [1, seq_len, num_classes, 1] -> [1, seq_len, num_classes]
            preds = preds.squeeze(-1)

        output = self.ctc_postprocess(preds)
        return output

    def setup_model(self):
        platform = "rknn"
        from rknn_executor import RKNN_model_container

        model = RKNN_model_container(self.model_path, self.target, self.device_id)
        return model, platform

    def release(self):
        self.model.release()


if __name__ == "__main__":
    det_model = TextRecognizer(target="rk3588", device_id=RKNNLite.NPU_CORE_0)

    # Set inputs
    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (REC_INPUT_SHAPE[1], REC_INPUT_SHAPE[0]))

    # Inference
    time1 = time.perf_counter()
    output = det_model.run(img)
    time2 = time.perf_counter()
    print(f"total inference time: {time2 - time1:.4f} s")

    print(output)
