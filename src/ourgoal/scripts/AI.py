#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
from pyzbar import pyzbar
import requests
import threading
import _thread as thread
import base64
import hashlib
import hmac
import json
import ssl
import websocket
import re
import os
import time
from datetime import datetime
from time import mktime
from urllib.parse import urlparse, urlencode
from wsgiref.handlers import format_date_time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ================= 1. 配置区 =================
MIC_DEVICE = "hw:XFMDPV0018"
APPID = 'f4ea634b'
APISecret = 'ZTA4YzI4NzNkNGE0NjVjODdiNWI5YjZm'
APIKey = '1c7f09de8fd38d0aebbc11059fcba203'

# 全局共享变量
qr_urls_set = set()
qr_scan_finished = False
ACT_global = "食品"
SIM_global = "电子产品"
task1_data_ready = False


bridge = CvBridge()
latest_frame = None
frame_lock = threading.Lock()

def image_callback(msg):
    global latest_frame
    try:
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        with frame_lock:
            latest_frame = frame.copy()
    except Exception as e:
        rospy.logwarn(f"图像转换失败: {e}")
# ================= 讯飞/网络工具函数 =================
class Ws_Param(object):
    def __init__(self, Spark_url):
        self.host = urlparse(Spark_url).netloc
        self.path = urlparse(Spark_url).path
        self.Spark_url = Spark_url
    def create_url(self):
        date = format_date_time(mktime(datetime.now().timetuple()))
        signature_origin = f"host: {self.host}\ndate: {date}\nGET {self.path} HTTP/1.1"
        signature_sha = hmac.new(APISecret.encode('utf-8'), signature_origin.encode('utf-8'), hashlib.sha256).digest()
        signature_sha_base64 = base64.b64encode(signature_sha).decode('utf-8')
        authorization_origin = f'api_key="{APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
        v = {"authorization": base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8'), "date": date, "host": self.host}
        return self.Spark_url + '?' + urlencode(v)

def get_iat_text(audio_path="/tmp/cmd.wav"):
    global iat_result_text
    iat_result_text = ""
    def on_message(ws, message):
        global iat_result_text
        try:
            data = json.loads(message)
            if data["code"] == 0:
                for w in data["data"]["result"]["ws"]:
                    for cw in w["cw"]: iat_result_text += cw["w"]
        except: pass
    def on_open(ws):
        def run(*args):
            status = 0
            with open(audio_path, "rb") as fp:
                while True:
                    buf = fp.read(8000)
                    if not buf: status = 2
                    d = {"data": {"status": status, "format": "audio/L16;rate=16000", "audio": str(base64.b64encode(buf), 'utf-8'), "encoding": "raw"}}
                    if status == 0: d.update({"common": {"app_id": APPID}, "business": {"domain": "iat", "language": "zh_cn", "accent": "mandarin", "vad_eos": 10000}})
                    ws.send(json.dumps(d))
                    if status == 2: break
                    time.sleep(0.04)
            time.sleep(1)
            ws.close()
        thread.start_new_thread(run, ())
    wsParam = Ws_Param("wss://iat-api.xfyun.cn/v2/iat")
    ws = websocket.WebSocketApp(wsParam.create_url(), on_message=on_message, on_open=on_open)
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    return iat_result_text

def get_spark_llm(prompt):
    global spark_response
    spark_response = ""
    def on_message(ws, message):
        global spark_response
        data = json.loads(message)
        if data['header']['code'] == 0:
            texts = data["payload"]["choices"].get("text", [])
            if texts: spark_response += texts[0].get("content", "")
            if data["payload"]["choices"]["status"] == 2: ws.close()
    def on_open(ws):
        def run(*args):
            data = {"header": {"app_id": APPID, "uid": "robot_car"}, "parameter": {"chat": {"domain": "spark-x", "temperature": 0.1, "max_tokens": 512}}, "payload": {"message": {"text": [{"role": "user", "content": prompt}]}}}
            ws.send(json.dumps(data))
        thread.start_new_thread(run, ())
    wsParam = Ws_Param("wss://spark-api.xf-yun.com/x2")
    ws = websocket.WebSocketApp(wsParam.create_url(), on_message=on_message, on_open=on_open)
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    try:
        json_str = re.search(r'\{.*\}', spark_response, re.DOTALL).group(0)
        return json.loads(json_str)
    except: return None

# ================= 业务工具 =================
def record_audio(duration=10, filename="/tmp/cmd.wav"):
    rospy.loginfo(f"🎤 录音开始 ({duration}s)...")
    os.system(f"arecord -D {MIC_DEVICE} -r 16000 -f S16_LE -c 1 -d {duration} {filename} -q")

def play_offline_tts(text):
    rospy.loginfo(f"🔊 播报: {text}")
    os.system(f'espeak -v zh+f2 "{text}" -s 130')

def fetch_items_from_urls(urls):
    items = []
    for url in urls:
        try:
            res = requests.get(url, timeout=3).json()
            if res.get("code") == 200: items.append(res.get("result"))
        except: pass
    return items

# ================= 后台逻辑线程 =================
def logic_worker():
    global ACT_global, SIM_global, qr_urls_set, qr_scan_finished

    # 1. 等待唤醒信号
    while rospy.get_param("awake", 0) == 0:
        time.sleep(0.1)
    rospy.set_param("awake", 0)

    # 2. 录音
    record_audio(10)
    
    # 3. 录音结束立刻放行底盘走迷宫 (C++ 看到 awake2 就开始跑)
    rospy.set_param("awake2", 1) 
    rospy.loginfo("🚀 录音完成，底盘已放行！后台开始解析指令...")

    # 4. 后台进行语音听写与大模型提取
    user_speech = get_iat_text("/tmp/cmd.wav")
    intent_res = get_spark_llm(f"严格分析指令：“{user_speech}”。任务1是实体抓取，任务2是仿真抓取。请精确提取这两个大类（只能是：食品/日用品/电子产品）。返回严格的JSON格式：{{\"ACT\":\"[实体大类]\", \"SIM\":\"[仿真大类]\"}}")
    ACT_global = intent_res.get("ACT", "食品") if intent_res else "食品"
    SIM_global = intent_res.get("SIM", "电子产品") if intent_res else "电子产品"
    rospy.loginfo(f"🧠 大类识别完成: 实体={ACT_global}, 仿真={SIM_global}")

    # 5. 等待小车到达 B 点信号
    rospy.loginfo("⏳ 等待底盘到达 B 点以进行最终匹配...")
    while rospy.get_param("start_qr_scan", 0) == 0:
        time.sleep(0.1)
    rospy.set_param("start_qr_scan", 0)

    # 6. 等待视觉扫齐 3 个码（如果在路上已经扫齐了，这里会直接跳过）
    while len(qr_urls_set) < 3 and not rospy.is_shutdown():
        time.sleep(0.1)
    
    qr_scan_finished = True
    rospy.set_param("qr_scan_done", 1) # 通知 C++ 停止旋转

    # 7. 第二次大模型匹配
    items = fetch_items_from_urls(list(qr_urls_set))
    allocate_res = get_spark_llm(f"物品：{items}。目标：实体={ACT_global}，仿真={SIM_global}。返回JSON：{{\"A1\":\"实体物品\", \"A2\":\"{ACT_global}\", \"A3\":\"实体车间\", \"B1\":\"仿真物品\", \"B2\":\"{SIM_global}\", \"B3\":\"仿真车间\"}}")

    if allocate_res:
        # 上传关键参数给 switch_test2.cpp
        rospy.set_param("real_item", allocate_res.get('A1', 'UNKNOWN'))
        rospy.set_param("real_class", ACT_global)
        rospy.set_param("real_room", allocate_res.get('A3', 'UNKNOWN'))

        rospy.set_param("sim_item", allocate_res.get('B1', 'UNKNOWN'))
        rospy.set_param("sim_class", SIM_global)
        rospy.set_param("sim_room", allocate_res.get('B3', 'UNKNOWN'))
        
        # 播报 1：识别完二维码后的语音
        play_offline_tts(f"取得{allocate_res['A1']}属于{allocate_res['A2']}应放置在{allocate_res['A3']}，仿真环境中取得{allocate_res['B1']}属于{allocate_res['B2']}应放置在{allocate_res['B3']}。")

    # 8. 任务 1 彻底结束，自杀释放 CPU
    rospy.set_param("task1_all_done", 1)
    rospy.loginfo("💀 任务1流程结束，AI 节点正在退出...")
    os.system("rosnode kill /smart_car_ai_node")

# ================= 主线程：视觉采集与扫码 =================
def main():
    global qr_urls_set, qr_scan_finished
    rospy.init_node('smart_car_ai_node', anonymous=True)

    t = threading.Thread(target=logic_worker)
    t.daemon = True
    t.start()

    rospy.loginfo("📷 视觉线程已启动，订阅 /ucar_camera/image_raw...")
    rospy.Subscriber("/ucar_camera/image_raw", Image, image_callback, queue_size=1)

    while not qr_scan_finished and not rospy.is_shutdown():
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            rospy.logwarn_throttle(2.0, "等待 /ucar_camera/image_raw 图像...")
            time.sleep(0.05)
            continue

        cv2.imshow("Always-On QR Scanner", frame)
        cv2.waitKey(1)

        for obj in pyzbar.decode(frame):
            url = obj.data.decode('utf-8')
            if url.startswith("http") and url not in qr_urls_set:
                qr_urls_set.add(url)
                rospy.loginfo(f"✅ 捕获新二维码 ({len(qr_urls_set)}/3): {url}")

        if len(qr_urls_set) >= 3:
            rospy.set_param("qr_scan_done", 1)

    cv2.destroyAllWindows()

    rospy.loginfo("🛑 摄像头扫描结束，主线程挂起，等待后台大模型处理与播报...")
    while not rospy.is_shutdown():
        time.sleep(1)

if __name__ == "__main__":
    main()
