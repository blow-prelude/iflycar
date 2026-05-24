import time

import cv2
import numpy as np
from paddleocr import TextDetection, TextRecognition

img_path = "/home/wtr/program/iflycar/src/find_signal/scripts/test2.jpg"
pic = cv2.imread("/home/wtr/program/iflycar/src/find_signal/scripts/test2.jpg")
pic = cv2.resize(pic, (160, 120))

# 文本检测，获取文本框
dect_model = TextDetection(
    model_name="PP-OCRv4_mobile_det",
    engine="paddle_static",
    engine_config={"cpu_threads": 2, "run_mode": "mkldnn"},
    thresh=0.7,
    box_thresh=0.7,
)
print("start to detect char...")
time1 = time.perf_counter()
output = dect_model.predict(pic, batch_size=1)
print(f"detect cost {time.perf_counter() - time1:6f} \n")

# 选择置信度最大的矩形框
best_res = max(output, key=lambda r: r["dt_scores"][0])
best_res.print()

x, y, w, h = cv2.boundingRect(best_res["dt_polys"][0].astype(np.int32))
cropped = pic[y : y + h, x : x + w]
cv2.imwrite("./find_signal/output/cropped.jpg", cropped)
# cv2.imshow("cropped", cropped)
# cv2.waitKey(0)


# 文本识别
recg_model = TextRecognition(
    model_name="PP-OCRv4_mobile_rec",
    engine="paddle_static",
    engine_config={"cpu_threads": 2, "run_mode": "mkldnn", "mkldnn_cache_capacity": 30},
)


print("start to recognize...")
time1 = time.perf_counter()
result = recg_model.predict(input=cropped)
print(f"predict cost {time.perf_counter() - time1:6f} \n")
for res in result:
    res.print()
    res.save_to_json("./find_signal/output/res.json")
