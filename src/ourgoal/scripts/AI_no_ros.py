#!/home/ucar/venv3.9/bin/python3
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import subprocess
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from time import mktime
from urllib.parse import urlencode, urlparse
from wsgiref.handlers import format_date_time

import websocket


# ================= 配置区 =================
MIC_DEVICE = "hw:XFMDPV0018"
RECORD_SECONDS = 12
WAV_DIR = Path(__file__).resolve().parents[1] / "wav"
PIPER_DIR = Path(__file__).resolve().parents[3] / "3rdparty" / "piper"
PIPER_EXECUTABLE = PIPER_DIR / "piper"
PIPER_MODEL = PIPER_DIR / "models" / "zh_CN-huayan-medium.onnx"
PIPER_CONFIG = PIPER_DIR / "models" / "zh_CN-huayan-medium.onnx.json"
PIPER_WAV = Path("/tmp/ucar_piper_ai_no_ros.wav")

APPID = os.environ.get("IFLYTEK_APPID", "f4ea634b")
API_SECRET = os.environ.get("IFLYTEK_API_SECRET", "ZTA4YzI4NzNkNGE0NjVjODdiNWI5YjZm")
API_KEY = os.environ.get("IFLYTEK_API_KEY", "1c7f09de8fd38d0aebbc11059fcba203")

ALLOWED_CATEGORIES = ("食品", "日用品", "电子产品")

# 第二次大模型匹配的候选物品直接维护在这里，不再扫描二维码或请求物品接口。
# 如果现场物品清单不同，只需要修改这个字典。
ITEMS_BY_CATEGORY = {
    "食品": ["苹果", "香蕉", "西瓜", "辣椒", "西红柿", "土豆", "牛奶", "蛋糕", "可乐"],
    "日用品": ["水杯", "牙刷", "毛巾"],
    "电子产品": ["手机", "耳机", "鼠标"],
}


LOGGER = logging.getLogger("ai_no_ros")


class WsParam:
    """讯飞 WebSocket 鉴权参数。"""

    def __init__(self, service_url):
        parsed_url = urlparse(service_url)
        self.host = parsed_url.netloc
        self.path = parsed_url.path
        self.service_url = service_url

    def create_url(self):
        date = format_date_time(mktime(datetime.now().timetuple()))
        signature_origin = (
            f"host: {self.host}\n"
            f"date: {date}\n"
            f"GET {self.path} HTTP/1.1"
        )
        signature = hmac.new(
            API_SECRET.encode("utf-8"),
            signature_origin.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        authorization_origin = (
            f'api_key="{API_KEY}", algorithm="hmac-sha256", '
            f'headers="host date request-line", '
            f'signature="{base64.b64encode(signature).decode("utf-8")}"'
        )
        params = {
            "authorization": base64.b64encode(
                authorization_origin.encode("utf-8")
            ).decode("utf-8"),
            "date": date,
            "host": self.host,
        }
        return f"{self.service_url}?{urlencode(params)}"


def record_audio():
    """启动后立即录音，并将 WAV 保存到 src/ourgoal/wav。"""
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    filename = WAV_DIR / f"command_{datetime.now():%Y%m%d_%H%M%S}.wav"
    LOGGER.info("开始录音（%ss）：%s", RECORD_SECONDS, filename)
    result = subprocess.run(
        [
            "arecord",
            "-D",
            MIC_DEVICE,
            "-r",
            "16000",
            "-f",
            "S16_LE",
            "-c",
            "1",
            "-d",
            str(RECORD_SECONDS),
            str(filename),
            "-q",
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"录音失败，arecord 退出码：{result.returncode}")
    return filename


def get_iat_text(audio_path):
    """调用讯飞实时语音听写，将 WAV 中的 PCM 数据发送给服务。"""
    result_text = []

    def on_message(ws, message):
        try:
            data = json.loads(message)
            if data.get("code") != 0:
                LOGGER.error("语音听写失败：%s", data)
                return
            for word in data.get("data", {}).get("result", {}).get("ws", []):
                result_text.extend(chunk.get("w", "") for chunk in word.get("cw", []))
        except (TypeError, ValueError, KeyError) as exc:
            LOGGER.warning("无法解析语音听写响应：%s", exc)

    def on_error(_ws, error):
        LOGGER.error("语音听写 WebSocket 错误：%s", error)

    def on_open(ws):
        def send_audio():
            try:
                with wave.open(str(audio_path), "rb") as audio_file:
                    first_frame = True
                    while True:
                        audio = audio_file.readframes(4000)
                        if not audio:
                            break
                        data = {
                            "data": {
                                "status": 0 if first_frame else 1,
                                "format": "audio/L16;rate=16000",
                                "audio": base64.b64encode(audio).decode("utf-8"),
                                "encoding": "raw",
                            }
                        }
                        if first_frame:
                            data["common"] = {"app_id": APPID}
                            data["business"] = {
                                "domain": "iat",
                                "language": "zh_cn",
                                "accent": "mandarin",
                                "vad_eos": 10000,
                            }
                            first_frame = False
                        ws.send(json.dumps(data, ensure_ascii=False))
                        time.sleep(0.04)

                ws.send(json.dumps({"data": {"status": 2, "audio": ""}}))
                time.sleep(1)
                ws.close()
            except Exception as exc:
                LOGGER.exception("发送语音数据失败：%s", exc)
                ws.close()

        threading.Thread(target=send_audio, daemon=True).start()

    ws_param = WsParam("wss://iat-api.xfyun.cn/v2/iat")
    websocket_app = websocket.WebSocketApp(
        ws_param.create_url(),
        on_message=on_message,
        on_error=on_error,
        on_open=on_open,
    )
    websocket_app.run_forever(sslopt={"cert_reqs": 0})
    return "".join(result_text)


def extract_json(text, call_name="未命名调用"):
    """从大模型可能附带 Markdown 的响应中提取 JSON 对象。"""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        LOGGER.error("星火响应中未找到 JSON 对象[%s]：%s", call_name, (text or "")[:1000])
        return None
    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        LOGGER.error(
            "星火响应 JSON 解析失败[%s]：%s，原始内容=%s",
            call_name,
            exc,
            (text or "")[:1000],
        )
        return None
    if not isinstance(result, dict):
        LOGGER.error("星火响应 JSON 不是对象[%s]：%s", call_name, (text or "")[:1000])
        return None
    return result


def get_spark_llm(prompt, call_name="未命名调用"):
    response_text = []

    def on_message(ws, message):
        try:
            data = json.loads(message)
            header = data.get("header", {})
            if header.get("code") != 0:
                LOGGER.error(
                    "星火大模型调用失败[%s]：code=%s，sid=%s，message=%s，原始响应=%s",
                    call_name,
                    header.get("code"),
                    header.get("sid"),
                    header.get("message"),
                    message[:1000],
                )
                ws.close()
                return
            choices = data.get("payload", {}).get("choices", {})
            for text in choices.get("text", []):
                response_text.append(text.get("content", ""))
            if choices.get("status") == 2:
                ws.close()
        except Exception as exc:
            LOGGER.error(
                "解析星火响应失败[%s]：%s，原始响应=%s",
                call_name,
                exc,
                message[:1000],
            )
            ws.close()

    def on_error(_ws, error):
        LOGGER.error("星火 WebSocket 异常[%s]：%s", call_name, error)

    def on_open(ws):
        try:
            request = {
                "header": {"app_id": APPID, "uid": "standalone_ai"},
                "parameter": {
                    "chat": {
                        "domain": "spark-x",
                        "temperature": 0.1,
                        "max_tokens": 512,
                    }
                },
                "payload": {
                    "message": {"text": [{"role": "user", "content": prompt}]}
                },
            }
            ws.send(json.dumps(request, ensure_ascii=False))
        except Exception as exc:
            LOGGER.error("发送星火请求失败[%s]：%s", call_name, exc)
            ws.close()

    ws_param = WsParam("wss://spark-api.xf-yun.com/x2")
    websocket_app = websocket.WebSocketApp(
        ws_param.create_url(),
        on_message=on_message,
        on_error=on_error,
        on_open=on_open,
    )
    try:
        websocket_app.run_forever(sslopt={"cert_reqs": 0})
    except Exception as exc:
        LOGGER.error("星火 WebSocket 运行失败[%s]：%s", call_name, exc)
        return None

    response = "".join(response_text)
    if not response:
        LOGGER.error("星火大模型未返回有效内容[%s]", call_name)
        return None
    return extract_json(response, call_name)


def normalize_category(value, default):
    value = str(value or "")
    return next(
        (category for category in ALLOWED_CATEGORIES if category in value),
        default,
    )


def match_items(act_category, sim_category):
    """让第二次大模型从代码中的固定清单选择两个物品。"""
    item_catalog = [
        {"name": item, "category": category}
        for category, items in ITEMS_BY_CATEGORY.items()
        for item in items
    ]
    prompt = (
        "请根据目标大类，从固定物品清单中分别选择一个实体物品和一个仿真物品。"
        "不得创造清单以外的物品。"
        f"固定物品清单：{json.dumps(item_catalog, ensure_ascii=False)}。"
        f"实体目标大类：{act_category}；仿真目标大类：{sim_category}。"
        "车间名称必须分别是“食品车间”“日用品车间”或“电子产品车间”。"
        "只返回严格 JSON，不要 Markdown："
        "{\"A1\":\"实体物品\",\"A2\":\"实体大类\","
        "\"A3\":\"实体车间\",\"B1\":\"仿真物品\","
        "\"B2\":\"仿真大类\",\"B3\":\"仿真车间\"}"
    )
    result = get_spark_llm(prompt, call_name="物品分配")
    if not result:
        return None

    valid_real_items = ITEMS_BY_CATEGORY[act_category]
    valid_sim_items = ITEMS_BY_CATEGORY[sim_category]
    real_item = result.get("A1")
    sim_item = result.get("B1")
    if real_item not in valid_real_items:
        LOGGER.warning("模型返回的实体物品不在清单中，使用清单首项：%s", valid_real_items[0])
        real_item = valid_real_items[0]
    if sim_item not in valid_sim_items:
        LOGGER.warning("模型返回的仿真物品不在清单中，使用清单首项：%s", valid_sim_items[0])
        sim_item = valid_sim_items[0]

    return {
        "A1": real_item,
        "A2": act_category,
        "A3": f"{act_category}车间",
        "B1": sim_item,
        "B2": sim_category,
        "B3": f"{sim_category}车间",
    }


def play_offline_tts(text):
    LOGGER.info("播报：%s", text)
    text = str(text)

    try:
        result = subprocess.run(
            [
                str(PIPER_EXECUTABLE),
                "--model", str(PIPER_MODEL),
                "--config", str(PIPER_CONFIG),
                "--output_file", str(PIPER_WAV),
            ],
            input=text + "\n",
            text=True,
            encoding="utf-8",
            cwd=str(PIPER_DIR),
            check=False,
        )
        if result.returncode == 0:
            result = subprocess.run(
                ["aplay", "--quiet", str(PIPER_WAV)],
                check=False,
            )
            if result.returncode == 0:
                return
            LOGGER.warning("Piper 音频播放失败，aplay 退出码：%s", result.returncode)
        else:
            LOGGER.warning("Piper 合成失败，退出码：%s", result.returncode)
    except OSError as exc:
        LOGGER.warning("Piper 播报启动失败：%s", exc)

    try:
        result = subprocess.run(
            ["espeak", "-v", "zh+f2", text, "-s", "130"],
            check=False,
        )
    except OSError as exc:
        LOGGER.warning("espeak 降级播报启动失败：%s", exc)
        return

    if result.returncode != 0:
        LOGGER.warning("语音播报失败，espeak 退出码：%s", result.returncode)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        # audio_path = record_audio()
        speech = get_iat_text("/tmp/cmd.wav")  # 直接使用预录音频文件，避免现场录音失败
        if not speech:
            LOGGER.error("语音识别失败或结果为空")
            return 1
        speech = speech.strip()
        if not speech:
            LOGGER.error("语音识别结果为空")
            return 1
        LOGGER.info("语音识别结果：%s", speech or "（空）")

        intent = get_spark_llm(
            f"严格分析这条语音指令：“{speech}”。任务1是实体抓取，任务2是仿真抓取。"
            "请提取两个大类，只能是：食品/日用品/电子产品。"
            "只返回严格 JSON：{\"ACT\":\"实体大类\",\"SIM\":\"仿真大类\"}",
            call_name="大类识别",
        ) or {}
        act_category = normalize_category(intent.get("ACT"), "食品")
        sim_category = normalize_category(intent.get("SIM"), "电子产品")
        LOGGER.info("大类识别完成：实体=%s，仿真=%s", act_category, sim_category)

        allocation = match_items(act_category, sim_category)
        if allocation is None:
            LOGGER.error("第二次大模型匹配失败")
            return 1

        print(json.dumps(allocation, ensure_ascii=False, indent=2))
        play_offline_tts(
            f"取得{allocation['A1']}属于{allocation['A2']}应放置在{allocation['A3']}，"
            f"仿真环境中取得{allocation['B1']}属于{allocation['B2']}应放置在{allocation['B3']}。"
        )
        return 0
    except KeyboardInterrupt:
        LOGGER.info("用户中断任务")
        return 130
    except Exception:
        LOGGER.exception("AI 任务执行失败")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
