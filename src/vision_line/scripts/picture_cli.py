import logging
import math
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
        self.send_threads = []
        self.running = None
        self.img_queues = None  # 改为队列列表
        self.num_images = None  # 添加图片数量字段
        self._is_closed = False

    def connect(self):
        """创建UDP socket（无需连接）"""
        if self._is_closed:
            raise RuntimeError("Sender is already closed.")

        self.cli_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logging.info(f"UDP socket created for {self.host}:{self.port}")

    def send_picture(
        self, img: cv2.typing.MatLike, img_id: int = 0, img_name: str = ""
    ) -> None:
        """通过UDP发送单张图片

        Args:
            img: OpenCV图像
            img_id: 图片ID，用于网格位置索引
            img_name: 图像名称，可打印ASCII，最长8字符

        Raises:
            ValueError: 如果 img_id 不在 0-255 范围内，或 img_name 不合法
        """
        if not (0 <= img_id <= 255):
            raise ValueError(f"img_id must be in range 0-255, got {img_id}")

        if not all(0x20 <= ord(c) <= 0x7E for c in img_name):
            raise ValueError(f"img_name must be printable ASCII, got {img_name!r}")
        if len(img_name) > 8:
            raise ValueError(f"img_name must be <= 8 chars, got {len(img_name)}")

        if self.cli_socket is None:
            raise RuntimeError("Socket is not initialized.")

        # 编码图片（320x240, JPEG质量70）
        res, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not res:
            raise RuntimeError("Failed to encode picture")

        # 新协议: [1字节ID][8字节名称][4字节大小][图片数据]
        name_bytes = img_name.encode("ascii").ljust(8, b"\x00")
        data = (
            img_id.to_bytes(1, "big")
            + name_bytes
            + len(buf).to_bytes(4, "big")
            + buf.tobytes()
        )

        # UDP发送
        try:
            self.cli_socket.sendto(data, (self.host, self.port))
            logging.debug(
                f"Picture sent with ID={img_id}, name={img_name!r}, size={len(buf)}"
            )
        except OSError as e:
            logging.error(f"UDP send failed: {e}")
            raise

    def _send_worker(
        self, thread_queues: list[tuple[int, Queue]], running_flag: list[bool]
    ) -> None:
        """发送线程工作函数：轮询分配的队列并发送

        Args:
            thread_queues: 分配给此线程的 (img_id, Queue) 列表
            running_flag: 运行标志列表，running_flag[0]为True时继续运行
        """
        thread_t_log = 0
        thread_cur_t, thread_pre_t, thread_t_sum = 0.0, 0.0, 0.0
        while running_flag[0]:
            try:
                thread_cur_t = time.perf_counter()
                thread_t_log += 1
                thread_t_sum += thread_cur_t - thread_pre_t
                if thread_t_log % 10 == 0:
                    thread_t_log = 0
                    logging.debug(
                        f"send thread frequence:{1 / (thread_t_sum / 10):.6f} Hz"
                    )
                    thread_t_sum = 0.0
                thread_pre_t = thread_cur_t

                any_sent = False
                for img_id, queue in thread_queues:
                    if not queue.empty():
                        try:
                            img, img_name = queue.get_nowait()
                            if img is not None:
                                self.send_picture(img, img_id, img_name)
                            queue.task_done()
                            any_sent = True
                        except Empty:
                            pass

                if not any_sent:
                    time.sleep(0.001)

            except Exception as e:
                logging.error(f"Error in send worker: {e}")
                continue

    def start_sending(
        self, num_images: int = 3, queue_size: int = 45, images_per_thread: int = None
    ) -> None:
        """启动发送线程

        Args:
            num_images: 图片数量（同时发送的图像流数量）
            queue_size: 每个队列的大小
            images_per_thread: 每个线程处理的图片数，默认等于num_images（单线程）

        Raises:
            RuntimeError: 如果发送器已关闭或未连接
            ValueError: 如果num_images或images_per_thread不合法
        """
        if self._is_closed:
            raise RuntimeError("Sender is already closed.")

        if self.cli_socket is None:
            raise RuntimeError("Not connected to server.")

        if not (1 <= num_images <= 255):
            raise ValueError(f"num_images must be in range 1-255, got {num_images}")

        ipt = images_per_thread if images_per_thread is not None else num_images
        if not isinstance(ipt, int) or ipt < 1:
            raise ValueError(f"images_per_thread must be a positive integer, got {ipt}")

        self.num_images = num_images
        self.img_queues = [Queue(maxsize=queue_size) for _ in range(num_images)]
        self.running = [True]

        num_threads = math.ceil(num_images / ipt)
        self.send_threads = []
        for t in range(num_threads):
            start_id = t * ipt
            end_id = min((t + 1) * ipt, num_images)
            thread_queues = [(i, self.img_queues[i]) for i in range(start_id, end_id)]
            thread = threading.Thread(
                target=self._send_worker, args=(thread_queues, self.running)
            )
            thread.daemon = True
            thread.start()
            self.send_threads.append(thread)
        logging.info(f"{num_threads} send thread(s) started for {num_images} images.")

    def enqueue_image(
        self, img: cv2.typing.MatLike, img_id: int = 0, img_name: str = ""
    ) -> None:
        """将图片放入对应ID的发送队列

        Args:
            img: OpenCV图像
            img_id: 图片队列ID (0 到 num_images-1)
            img_name: 图像名称
        """
        if self.img_queues is None:
            raise RuntimeError("Sender is not started.")

        if img_id < 0 or img_id >= self.num_images:
            raise RuntimeError(
                f"Invalid img_id: {img_id}, must be 0-{self.num_images - 1}"
            )

        if not self.img_queues[img_id].full():
            self.img_queues[img_id].put((img, img_name))
        else:
            logging.debug(f"Queue {img_id} full, dropping frame.")

    def close(self):
        """关闭发送器，释放所有资源"""
        if self._is_closed:
            return

        self._is_closed = True

        if self.running is not None:
            self.running[0] = False

        for thread in self.send_threads:
            thread.join(timeout=2)
        self.send_threads = []

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
    host = "192.168.10.105"
    port = 12345
    main_t_log = 0
    main_t_sum = 0
    main_pre_t = 0.0

    try:
        with CameraCapture(0) as camera, ImageSender(host, port) as sender:
            sender.connect()
            sender.start_sending(num_images=5, images_per_thread=2)  # 3个发送线程

            # cap = cv2.VideoCapture("test2.avi")

            try:
                while True:
                    main_cur_t = time.perf_counter()
                    main_t_sum += main_cur_t - main_pre_t
                    main_t_log += 1
                    if main_t_log % 10 == 0:
                        main_t_log = 0
                        logging.debug(
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
                    img3 = img.copy()
                    img4 = img.copy()

                    sender.enqueue_image(img0, img_id=0, img_name="img0")
                    sender.enqueue_image(img1, img_id=1, img_name="img1")
                    sender.enqueue_image(img2, img_id=2, img_name="img2")
                    sender.enqueue_image(img3, img_id=3, img_name="img3")
                    sender.enqueue_image(img4, img_id=4, img_name="img4")

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
