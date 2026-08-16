#!/home/ucar/venv3.9/bin/python3
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
import subprocess
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

bridge = CvBridge()
latest_frame = None
frame_lock = threading.Lock()
qr_lock = threading.Lock()
shutdown_event = threading.Event()


def handle_shutdown():
    """Wake worker waits promptly when ROS asks this node to stop."""
    shutdown_event.set()


def wait_for_param(param_name, expected_value=1, reset_value=None):
    """Wait for a ROS parameter while still allowing a clean shutdown."""
    while not rospy.is_shutdown() and not shutdown_event.is_set():
        if rospy.get_param(param_name, 0) == expected_value:
            if reset_value is not None:
                rospy.set_param(param_name, reset_value)
            return True
        shutdown_event.wait(0.1)
    return False


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

def get_spark_llm(prompt, call_name="未命名调用"):
    global spark_response
    spark_response = ""
    def on_message(ws, message):
        global spark_response
        try:
            data = json.loads(message)
            header = data.get("header", {})
            if header.get("code") != 0:
                rospy.logerr(
                    f"星火大模型调用失败[{call_name}]: "
                    f"code={header.get('code')}, sid={header.get('sid')}, "
                    f"message={header.get('message')}, 原始响应={message[:1000]}"
                )
                ws.close()
                return

            choices = data.get("payload", {}).get("choices", {})
            texts = choices.get("text", [])
            if texts:
                spark_response += texts[0].get("content", "")
            if choices.get("status") == 2:
                ws.close()
        except Exception as exc:
            rospy.logerr(
                f"解析星火响应失败[{call_name}]: {exc}, 原始响应={message[:1000]}"
            )
            ws.close()

    def on_error(ws, error):
        rospy.logerr(f"星火 WebSocket 异常[{call_name}]: {error}")

    def on_open(ws):
        def run(*args):
            try:
                data = {"header": {"app_id": APPID, "uid": "robot_car"}, "parameter": {"chat": {"domain": "spark-x", "temperature": 0.1, "max_tokens": 512}}, "payload": {"message": {"text": [{"role": "user", "content": prompt}]}}}
                ws.send(json.dumps(data))
            except Exception as exc:
                rospy.logerr(f"发送星火请求失败[{call_name}]: {exc}")
                ws.close()
        thread.start_new_thread(run, ())
    wsParam = Ws_Param("wss://spark-api.xf-yun.com/x2")
    ws = websocket.WebSocketApp(
        wsParam.create_url(),
        on_message=on_message,
        on_error=on_error,
        on_open=on_open,
    )
    try:
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    except Exception as exc:
        rospy.logerr(f"星火 WebSocket 运行失败[{call_name}]: {exc}")
        return None

    if not spark_response:
        rospy.logerr(f"星火大模型未返回有效内容[{call_name}]")
        return None

    match = re.search(r'\{.*\}', spark_response, re.DOTALL)
    if not match:
        rospy.logerr(
            f"星火响应中未找到 JSON 对象[{call_name}]: {spark_response[:1000]}"
        )
        return None

    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        rospy.logerr(
            f"星火响应 JSON 解析失败[{call_name}]: {exc}, "
            f"原始内容={spark_response[:1000]}"
        )
        return None

    if not isinstance(result, dict):
        rospy.logerr(
            f"星火响应 JSON 不是对象[{call_name}]: {spark_response[:1000]}"
        )
        return None
    return result

# ================= 业务工具 =================
def record_audio(duration=10, filename="/tmp/cmd.wav"):
    rospy.loginfo(f" 录音开始 ({duration}s)...")
    result = subprocess.run(
        ["arecord", "-D", MIC_DEVICE, "-r", "16000", "-f", "S16_LE",
         "-c", "1", "-d", str(duration), filename, "-q"],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"录音失败，arecord 退出码: {result.returncode}")

def play_offline_tts(text):
    rospy.loginfo(f" 播报: {text}")
    result = subprocess.run(
        ["espeak", "-v", "zh+f2", str(text), "-s", "130"],
        check=False,
    )
    if result.returncode != 0:
        rospy.logwarn(f"语音播报失败，espeak 退出码: {result.returncode}")

def fetch_items_from_urls(urls):
    items = []
    for url in urls:
        try:
            response = requests.get(url, timeout=3)
        except requests.RequestException as exc:
            rospy.logerr(f"请求二维码物品接口失败: url={url}, error={exc}")
            continue

        try:
            res = response.json()
        except ValueError as exc:
            rospy.logerr(
                f"二维码物品接口响应不是有效 JSON: url={url}, "
                f"status={response.status_code}, error={exc}"
            )
            continue

        if res.get("code") != 200:
            rospy.logerr(
                f"二维码物品接口返回失败: url={url}, status={response.status_code}, "
                f"response={str(res)[:1000]}"
            )
            continue
        items.append(res.get("result"))
    return items

# ================= 后台逻辑线程 =================
def logic_worker():
    global ACT_global, SIM_global, qr_urls_set, qr_scan_finished

    try:
        # 1. 阻塞等待唤醒信号
        if not wait_for_param("awake", reset_value=0):
            rospy.logwarn("阻塞等待唤醒信号超时或被中断，AI 逻辑线程退出。")
            return

        # 2. 录音
        record_audio(20)

        # 3. 录音结束立刻放行底盘走迷宫 (C++ 看到 awake2 就开始跑)
        rospy.set_param("awake2", 1)
        rospy.loginfo(" 录音完成，底盘已放行！后台开始解析指令...")

        # 4. 后台进行语音听写与大模型提取
        user_speech = get_iat_text("/tmp/cmd.wav")
        if not user_speech:
            rospy.logwarn("语音识别失败或结果为空，默认使用实体=食品，仿真=电子产品")
            ACT_global = "食品"
            SIM_global = "电子产品"
        else:
            user_speech = user_speech.strip()
            rospy.loginfo(f" 语音识别结果: {user_speech}")
            
        intent_res = get_spark_llm(
            "你是任务指令分类器。\n"
            f"待解析指令：{json.dumps(user_speech, ensure_ascii=False)}\n"
            "任务1是实体抓取，对应 ACT；任务2是仿真抓取，对应 SIM。\n"
            "必须遵守以下规则：\n"
            "1. ACT 和 SIM 的值只能是“食品”、“日用品”或“电子产品”。\n"
            "2. 根据指令中的任务顺序分别提取，不得交换 ACT 和 SIM。\n"
            "3. 若某个任务无法判断，ACT 使用“食品”，SIM 使用“电子产品”。\n"
            "4. 只输出一个 JSON 对象，不要输出 Markdown、解释或其他文字。\n"
            "输出格式：{\"ACT\":\"实体大类\",\"SIM\":\"仿真大类\"}",
            call_name="大类识别",
        )
        ACT_global = intent_res.get("ACT", "食品") if intent_res else "食品"
        SIM_global = intent_res.get("SIM", "电子产品") if intent_res else "电子产品"
        rospy.loginfo(f"🧠 大类识别完成: 实体={ACT_global}, 仿真={SIM_global}")

        # 5. 阻塞等待小车到达 B 点信号
        rospy.loginfo("⏳ 等待底盘到达 B 点以进行最终匹配...")
        if not wait_for_param("start_qr_scan", reset_value=0):
            rospy.logwarn("阻塞等待扫码信号超时或被中断，AI 逻辑线程退出。")
            return

        # 6. 等待视觉扫齐 3 个码（如果在路上已经扫齐了，这里会直接跳过）
        while not rospy.is_shutdown() and not shutdown_event.is_set():
            with qr_lock:
                if len(qr_urls_set) >= 3:
                    qr_urls = list(qr_urls_set)
                    break
            shutdown_event.wait(0.1)
        else:
            return

        qr_scan_finished = True
        rospy.set_param("qr_scan_done", 1)  # 通知 C++ 停止旋转

        # 7. 第二次大模型匹配
        items = fetch_items_from_urls(qr_urls)
        allocate_res = get_spark_llm(
            "你是物品分类与任务分配器。\n"
            f"候选物品（JSON）：{json.dumps(items, ensure_ascii=False)}\n"
            f"实体任务目标类别：{ACT_global}\n"
            f"仿真任务目标类别：{SIM_global}\n"
            "必须遵守以下规则：\n"
            "1. A1 和 B1 必须是候选物品中的原始物品名称，不得创造、改写或补充物品。\n"
            "2. A1 必须属于实体目标类别，B1 必须属于仿真目标类别。\n"
            f"3. A2 必须原样输出“{ACT_global}”，B2 必须原样输出“{SIM_global}”。\n"
            "4. 类别与车间的唯一映射为：食品→食品车间，日用品→日用品车间，"
            "电子产品→电子产品车间。\n"
            "5. A3 必须是 A2 对应的车间，B3 必须是 B2 对应的车间。"
            "禁止输出“物品名+生产车间”或任何其他车间名称。\n"
            "6. 只输出一个 JSON 对象，键名和顺序必须严格为 "
            "A1、A2、A3、B1、B2、B3，不要输出 Markdown、解释或其他文字。\n"
            "输出格式："
            "{\"A1\":\"实体物品\",\"A2\":\"实体大类\",\"A3\":\"实体车间\","
            "\"B1\":\"仿真物品\",\"B2\":\"仿真大类\",\"B3\":\"仿真车间\"}",
            call_name="物品分配",
        )

        if allocate_res:
            # 上传关键参数给 switch_test2.cpp
            real_item = allocate_res.get("A1", "UNKNOWN")
            real_class = allocate_res.get("A2", ACT_global)
            real_room = allocate_res.get("A3", "UNKNOWN")
            sim_item = allocate_res.get("B1", "UNKNOWN")
            sim_class = allocate_res.get("B2", SIM_global)
            sim_room = allocate_res.get("B3", "UNKNOWN")

            rospy.set_param("real_item", real_item)
            rospy.set_param("real_class", ACT_global)
            rospy.set_param("real_room", real_room)
            rospy.set_param("sim_item", sim_item)
            rospy.set_param("sim_class", SIM_global)
            rospy.set_param("sim_room", sim_room)

            # 播报 1：识别完二维码后的语音
            play_offline_tts(
                f"取得{real_item}属于{real_class}应放置在{real_room}，"
                f"仿真环境中取得{sim_item}属于{sim_class}应放置在{sim_room}。"
            )

        # 8. 任务 1 彻底结束，通知 ROS 正常停止本节点
        rospy.set_param("task1_all_done", 1)
        rospy.loginfo("✅ 任务1流程结束，AI 节点正在退出...")
        rospy.signal_shutdown("任务1已完成")
    except rospy.ROSInterruptException:
        return
    except Exception as exc:
        rospy.logerr(f"AI 任务流程异常: {exc}")
        rospy.signal_shutdown("AI 任务流程异常")

# ================= 主线程：视觉采集与扫码 =================
def main():
    global qr_urls_set, qr_scan_finished
    rospy.init_node('smart_car_ai_node', anonymous=True)
    rospy.on_shutdown(handle_shutdown)

    worker = threading.Thread(target=logic_worker, name="ai-logic-worker", daemon=True)
    worker.start()

    rospy.loginfo("📷 视觉线程已启动，订阅 /ucar_camera/image_raw...")
    subscriber = None

    try:
        subscriber = rospy.Subscriber(
            "/ucar_camera/image_raw", Image, image_callback, queue_size=1
        )

        while not qr_scan_finished and not rospy.is_shutdown():
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()

            if frame is None:
                rospy.logwarn_throttle(2.0, "等待 /ucar_camera/image_raw 图像...")
                shutdown_event.wait(0.05)
                continue

            cv2.imshow("Always-On QR Scanner", frame)
            cv2.waitKey(1)

            for obj in pyzbar.decode(frame):
                url = obj.data.decode("utf-8")
                if not url.startswith("http"):
                    continue
                with qr_lock:
                    if url in qr_urls_set:
                        continue
                    qr_urls_set.add(url)
                    qr_count = len(qr_urls_set)
                rospy.loginfo(f"✅ 捕获新二维码 ({qr_count}/3): {url}")
                if qr_count >= 3:
                    rospy.set_param("qr_scan_done", 1)

        if not rospy.is_shutdown():
            rospy.loginfo(" 摄像头扫描结束，等待后台大模型处理与播报...")

        while worker.is_alive() and not rospy.is_shutdown():
            worker.join(timeout=0.2)
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        rospy.logerr(f"AI 主线程异常: {exc}")
        rospy.signal_shutdown("AI 主线程异常")
    finally:
        shutdown_event.set()
        if subscriber is not None:
            try:
                subscriber.unregister()
            except Exception as exc:
                rospy.logwarn(f"注销图像订阅失败: {exc}")
        try:
            cv2.destroyAllWindows()
        except Exception as exc:
            rospy.logwarn(f"关闭 OpenCV 窗口失败: {exc}")
        worker.join(timeout=2.0)
        if worker.is_alive():
            rospy.logwarn("AI 逻辑线程未在 2 秒内结束，将随主进程退出")


if __name__ == "__main__":
    main()
