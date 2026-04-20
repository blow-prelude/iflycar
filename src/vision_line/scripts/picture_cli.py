import logging
import socket
import threading
import time
from queue import Empty, Queue

import cv2


class CameraConfig:
    INDEX = 0
    WIDTH = 320
    HEIGHT = 240


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
        self.img_queues = None  # 改为队列列表
        self.num_images = None  # 添加图片数量字段
        self._is_closed = False

        self.thread_t_log = 0
        self.thread_t_sum = 0

    def connect(self):
        """创建UDP socket（无需连接）"""
        if self._is_closed:
            raise RuntimeError("Sender is already closed.")

        self.cli_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logging.info(f"UDP socket created for {self.host}:{self.port}")

    def send_picture(self, img: cv2.typing.MatLike, img_id: int = 0) -> None:
        """通过UDP发送单张图片

        Args:
            img: OpenCV图像
            img_id: 图片ID，用于区分不同的图像流

        Raises:
            ValueError: 如果 img_id 不在 0-255 范围内
        """
        if not (0 <= img_id <= 255):
            raise ValueError(f"img_id must be in range 0-255, got {img_id}")

        if self.cli_socket is None:
            raise RuntimeError("Socket is not initialized.")

        # 编码图片（320x240, JPEG质量70）
        res, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not res:
            raise RuntimeError("Failed to encode picture")

        # 新协议: [1字节ID][4字节大小][图片数据]
        data = (
            img_id.to_bytes(1, "big")  # 图片ID
            + len(buf).to_bytes(4, "big")  # 图片大小
            + buf.tobytes()  # 图片内容
        )

        # UDP发送
        try:
            self.cli_socket.sendto(data, (self.host, self.port))
            logging.debug(f"Picture sent with ID={img_id}, size={len(buf)}")
        except OSError as e:
            logging.error(f"UDP send failed: {e}")
            raise

    def _send_worker(self, queues: list[Queue], running_flag: list[bool]) -> None:
        """发送线程工作函数：轮询所有队列并发送"""
        thread_t_log = 0
        thread_cur_t, thread_pre_t, thread_t_sum = 0.0, 0.0, 0.0
        while running_flag[0]:
            try:
                # 测速
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

                # 轮询所有队列
                any_sent = False
                for img_id, queue in enumerate(queues):
                    if not queue.empty():
                        try:
                            img = queue.get_nowait()
                            if img is not None:
                                self.send_picture(img, img_id)
                            queue.task_done()
                            any_sent = True
                        except Empty:
                            pass  # 队列为空继续

                if not any_sent:
                    time.sleep(0.001)  # 所有队列都为空时才休眠

            except Exception as e:
                logging.error(f"Error in send worker: {e}")
                continue

    def start_sending(self, num_images: int = 3, queue_size: int = 45) -> None:
        """启动发送线程

        Args:
            num_images: 图片数量（同时发送的图像流数量）
            queue_size: 每个队列的大小
        """
        if self._is_closed:
            raise RuntimeError("Sender is already closed.")

        if self.cli_socket is None:
            raise RuntimeError("Not connected to server.")

        if not (1 <= num_images <= 255):
            raise ValueError(f"num_images must be in range 1-255, got {num_images}")

        # 创建多个图片队列
        self.num_images = num_images
        self.img_queues = [Queue(maxsize=queue_size) for _ in range(num_images)]
        self.running = [True]

        # 启动发送线程
        self.send_thread = threading.Thread(
            target=self._send_worker, args=(self.img_queues, self.running)
        )
        self.send_thread.daemon = True
        self.send_thread.start()
        logging.info(f"Send thread started for {num_images} images.")

    def enqueue_image(self, img: cv2.typing.MatLike, img_id: int = 0) -> None:
        """将图片放入对应ID的发送队列

        Args:
            img: OpenCV图像
            img_id: 图片队列ID (0 到 num_images-1)
        """
        if self.img_queues is None:
            raise RuntimeError("Sender is not started.")

        if img_id < 0 or img_id >= self.num_images:
            raise RuntimeError(
                f"Invalid img_id: {img_id}, must be 0-{self.num_images - 1}"
            )

        if not self.img_queues[img_id].full():
            self.img_queues[img_id].put(img)
        else:
            logging.debug(f"Queue {img_id} full, dropping frame.")

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
    main_t_log = 0
    main_t_sum = 0
    main_pre_t = 0.0

    try:
        with CameraCapture(0) as camera, ImageSender(host, port) as sender:
            sender.connect()
            sender.start_sending(num_images=3)  # 启动3个图像流

            # cap = cv2.VideoCapture("test2.avi")

            try:
                while True:
                    main_cur_t = time.perf_counter()
                    main_t_sum += main_cur_t - main_pre_t
                    main_t_log += 1
                    if main_t_log % 10 == 0:
                        main_t_log = 0
                        logging.info(
                            f"main thread frequence:{1 / (main_t_sum / 10):6f} Hz"
                        )
                        main_t_sum = 0
                    main_pre_t = main_cur_t

                    # 获取图片并复制模拟多张图片
                    img = camera.get_picture()
                    # ret, img = cap.read()
                    if img is None:
                        break

                    # 复制原图模拟3张图片（实际使用时替换为真实处理逻辑）
                    img0 = img.copy()
                    img1 = img.copy()
                    img2 = img.copy()

                    sender.enqueue_image(img0, img_id=0)
                    sender.enqueue_image(img1, img_id=1)
                    sender.enqueue_image(img2, img_id=2)

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
