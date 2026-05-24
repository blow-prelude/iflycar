import os
import time

import cv2
from paddleocr import TextDetection, TextRecognition

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "test.jpg")
output_dir = os.path.join(os.path.dirname(script_dir), "output")
pic = cv2.imread(img_path)
pic = cv2.resize(pic, (320, 240))

# 文本检测，获取文本框
dect_model = TextDetection(
    model_name="PP-OCRv4_mobile_det",
    engine="paddle_static",
    engine_config={"cpu_threads": 2},
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
cv2.imwrite(os.path.join(output_dir, "cropped.jpg"), cropped)
# cv2.imshow("cropped", cropped)
# cv2.waitKey(0)


# 文本识别
recg_model = TextRecognition(
    model_name="PP-OCRv4_mobile_rec",
    engine="paddle_static",
    engine_config={"cpu_threads": 8},
)


print("start to recognize...")
time1 = time.perf_counter()
result = recg_model.predict(input=cropped)
print(f"predict cost {time.perf_counter() - time1:6f} \n")
for res in result:
    res.print()
    res.save_to_json(os.path.join(output_dir, "res.json"))
