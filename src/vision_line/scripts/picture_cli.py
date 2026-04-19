import logging
import socket
import threading
import time
from queue import Queue

import cv2


class CameraConfig:
    INDEX = 0
    WIDTH = 640
    HEIGHT = 480


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ImageSender:
    def __init__(self, host, port) -> None:
        self.host = host
        self.port = port
        self.cli_socket = None
        self.send_thread = None
        self.running = None
        self.img_queue = None
        self._is_closed = False

        self.thread_t_log = 0
        self.thread_t_sum = 0

    def connect(self):
        """连接到服务器"""
        if self._is_closed:
            raise RuntimeError("Sender is already closed.")

        self.cli_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.cli_socket.connect((self.host, self.port))
            logging.info(f"Connected to server at {self.host}:{self.port}")
        except Exception as e:
            self.cli_socket.close()
            self.cli_socket = None
            raise RuntimeError(f"Connection failed: {e}") from e

    def send_picture(self, img):
        """发送单张图片"""
        if self.cli_socket is None:
            raise RuntimeError("Socket is not connected.")

        # 编码
        res, buf = cv2.imencode(
            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )  # 按照jpeg格式编码图像，质量为原先的85
        if not res:
            raise RuntimeError("Failed to encode picture")

        data = (
            len(buf).to_bytes(4, "big") + buf.tobytes()
        )  # 二进制打包，前4个字节是图片大小，后面是图片内容

        # 发送图片
        self.cli_socket.sendall(data)
        logging.debug("Picture sent.")

    def _send_worker(self, queue, running_flag):
        """发送线程工作函数：从队列中取出图片并发送"""
        thread_t_log = 0
        thread_cur_t, thread_pre_t, thread_t_sum = 0.0, 0.0, 0.0
        while running_flag[0]:
            try:
                # 测速，每10轮计算一次发送频率
                thread_cur_t = time.perf_counter()
                thread_t_log += 1
                thread_t_sum += thread_cur_t - thread_pre_t
                if thread_t_log % 10 == 0:
                    thread_t_log = 0
                    logging.info(
                        f"send thread frequence:{1 / (thread_t_sum / 10):.6f} Hz"
                    )
                    thread_t_sum = 0.0
                thread_pre_t = thread_cur_t

                img = queue.get(timeout=0.04)  # 从队列获取图片，超时0.04秒就会抛出异常
                if img is not None:
                    self.send_picture(img)
                queue.task_done()  # 标记一个任务已经完成，减少一个引用记数(与之对应的，put()会增加一个引用记数)，当所有任务完成后，join()会阻塞直到引用记数为0

            except Exception:
                continue  # 队列为空时继续循环

    def start_sending(self, queue_size=45):
        """启动发送线程"""
        if self._is_closed:
            raise RuntimeError("Sender is already closed.")

        if self.cli_socket is None:
            raise RuntimeError("Not connected to server.")

        # 创建图片队列和发送线程
        self.img_queue = Queue(maxsize=queue_size)
        self.running = [True]  # 使用列表以便在线程间共享

        # 启动发送线程
        self.send_thread = threading.Thread(
            target=self._send_worker, args=(self.img_queue, self.running)
        )
        self.send_thread.daemon = True
        self.send_thread.start()
        logging.info("Send thread started.")

    def enqueue_image(self, img):
        """将图片放入发送队列"""
        if self.img_queue is None:
            raise RuntimeError("Sender is not started.")

        if not self.img_queue.full():
            self.img_queue.put(img)
        else:
            logging.debug("Queue full, dropping frame.")

    def close(self):
        """关闭发送器，释放所有资源"""
        if self._is_closed:
            return

        self._is_closed = True

        # 停止发送线程
        if self.running is not None:
            self.running[0] = False

        # 等待线程结束
        if self.send_thread is not None:
            self.send_thread.join(timeout=2)
            self.send_thread = None

        # 关闭套接字
        if self.cli_socket is not None:
            self.cli_socket.close()
            self.cli_socket = None
            logging.info("Sender closed.")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """上下文管理器出口，自动释放资源"""
        self.close()
        return False

    def __del__(self):
        """析构函数，对象销毁时自动释放资源"""
        self.close()


class CameraCapture:
    def __init__(self, index=None, width=None, height=None) -> None:
        """初始化摄像头

        Args:
            index: 摄像头索引，默认使用 CameraConfig.INDEX
            width: 分辨率宽度，默认使用 CameraConfig.WIDTH
            height: 分辨率高度，默认使用 CameraConfig.HEIGHT
        """
        idx = index if index is not None else CameraConfig.INDEX
        w = width if width is not None else CameraConfig.WIDTH
        h = height if height is not None else CameraConfig.HEIGHT

        # self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        self.cap = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        if not self.cap.isOpened():
            logging.error("Cannot open camera.")
            raise RuntimeError(f"Failed to open camera at index {idx}")

    def get_picture(self):
        """获取一帧图片"""
        ret, frame = self.cap.read()
        if not ret:
            logging.error("Failed to grab frame")
            return None
        return frame

    def is_opened(self):
        """检查摄像头是否已打开"""
        return self.cap is not None and self.cap.isOpened()

    def close(self):
        """关闭摄像头"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """上下文管理器出口，自动释放资源"""
        self.close()
        return False

    def __del__(self):
        """析构函数，对象销毁时自动释放资源"""
        self.close()


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 12345
    main_t_log = 0  # 测速，每10轮计算一次
    main_t_sum = 0
    main_pre_t = 0.0

    # 方式1: 使用上下文管理器
    # 自动管理摄像头和发送器的资源释放
    try:
        with CameraCapture(0) as camera, ImageSender(host, port) as sender:
            # 连接到服务器
            sender.connect()

            # 启动发送线程
            sender.start_sending(queue_size=45)

            try:
                while True:
                    main_cur_t = time.time()
                    main_t_sum += main_cur_t - main_pre_t
                    main_t_log += 1
                    if main_t_log % 10 == 0:
                        main_t_log = 0
                        logging.info(
                            f"main thread frequence:{1 / (main_t_sum / 10):6f} Hz"
                        )
                        main_t_sum = 0
                    main_pre_t = main_cur_t

                    # 获取图片
                    img = camera.get_picture()
                    if img is None:
                        break

                    # 显示本地画面
                    # cv2.imshow("raw_img", img)

                    # 将图片放入发送队列
                    sender.enqueue_image(img)

                    # 按 ESC 退出
                    if cv2.waitKey(20) & 0xFF == 27:
                        logging.info("User interrupted by ESC key.")
                        break

            except KeyboardInterrupt:
                logging.info("User interrupted by Ctrl+C.")

            finally:
                cv2.destroyAllWindows()
                logging.info("Resources released.")

    except RuntimeError as e:
        logging.error(f"Runtime error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
