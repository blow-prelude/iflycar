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
import argparse
import os

import cv2
import numpy as np
import operators
from rec_postprocess import CTCLabelDecode

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录
parent_dir = os.path.dirname(script_dir)

# 输入图像路径
IMG_PATH = os.path.join(script_dir, "test11.png")


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
    def __init__(self, args) -> None:
        self.model, self.framework = setup_model(args)
        self.preprocess_funct = []
        for item in PRE_PROCESS_CONFIG:
            for key in item:
                pclass = getattr(operators, key)
                p = pclass(**item[key])
                self.preprocess_funct.append(p)

        self.ctc_postprocess = CTCLabelDecode(**POSTPROCESS_CONFIG["CTCLabelDecode"])

    def preprocess(self, img):
        for p in self.preprocess_funct:
            img = p(img)

        if self.framework == "onnx":
            image_input = img["image"]
            image_input = image_input.reshape(1, *image_input.shape)
            image_input = image_input.transpose(0, 3, 1, 2)
            img["image"] = image_input
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
    parser = argparse.ArgumentParser(description="PPOCR-Rec Python Demo")
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
    det_model = TextRecognizer(args)

    # Set inputs
    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (REC_INPUT_SHAPE[1], REC_INPUT_SHAPE[0]))

    # Inference
    output = det_model.run(img)

    print(output)
