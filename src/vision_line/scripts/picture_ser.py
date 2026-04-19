import logging
import socket
import threading
import time

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ImageReceiver:
    def __init__(self, host, port) -> None:
        self.port = port
        self.host = host
        self.ser_socket = None
        self.cli_socket = None
        self.connect_thread = None
        self.running = None
        self._is_closed = False

        self.receive_thread_t_sum = 0
        self.receive_thread_t_log = 0

        # 创建监听套接字
        self.ser_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ser_socket.bind((self.host, self.port))
        self.ser_socket.listen(1)
        self.ser_socket.settimeout(1)  # 设置超时时间，避免一直阻塞在accept/recv
        logging.info(f"server start , listening port {self.port}...")

    def receive_picture(self):
        """一对一接收图片，接收方为server"""
        if self._is_closed:
            raise RuntimeError("Receiver is already closed.")

        if self.ser_socket is None:
            raise RuntimeError("Server socket is not initialized.")

        self.cli_socket = None
        self.connect_thread = None
        self.running = [True]  # 使用列表以便在线程间共享

        try:
            while True:
                try:
                    self.cli_socket, addr = self.ser_socket.accept()
                    logging.info(f"client {addr} connected, start receiving...")
                    break
                except socket.timeout:
                    continue  # 超时后继续等待连接
                except Exception as e:
                    raise RuntimeError(f"Error connecting to client: {e}") from e

            self.connect_thread = threading.Thread(
                target=self.receive_worker, args=(self.cli_socket, self.running)
            )
            self.connect_thread.daemon = True
            self.connect_thread.start()
            logging.info("start receive thread.")

            # 主线程等待，直到用户按下 ESC 或接收线程结束
            while self.connect_thread.is_alive():
                self.connect_thread.join(timeout=0.1)

        except KeyboardInterrupt:
            logging.info("\nshut down by user.")
            if self.running:
                self.running[0] = False  # 通知接收线程停止
        finally:
            self._cleanup_client()

    def _cleanup_client(self):
        """清理客户端连接和线程资源"""
        if self.connect_thread is not None:
            self.connect_thread.join(timeout=2)
            self.connect_thread = None
        if self.cli_socket is not None:
            self.cli_socket.close()
            self.cli_socket = None

    def close(self):
        """显式关闭服务器，释放所有资源"""
        if self._is_closed:
            return

        self._is_closed = True

        # 停止接收线程
        if self.running is not None:
            self.running[0] = False

        # 清理客户端资源
        self._cleanup_client()

        # 关闭服务器套接字
        if self.ser_socket is not None:
            try:
                self.ser_socket.shutdown(socket.SHUT_RDWR)
                logging.info("shut down connection...")
            except Exception:
                pass
            self.ser_socket.close()
            self.ser_socket = None
            logging.info("Server closed.")

    def __enter__(self):
        """上下文管理器入口，支持 with 语句"""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """上下文管理器出口，自动释放资源"""
        self.close()
        return False  # 不抑制异常

    def __del__(self):
        """析构函数，对象销毁时自动释放资源"""
        self.close()

    def handle_connect(self, conn):
        # 接收文件头，即图像大小
        header = conn.recv(4)
        if len(header) != 4:
            raise RuntimeError(f"Invalid header with {len(header)} bytes.")
        size = int.from_bytes(header, "big")  # 将4字节长的文件头转成整形数据
        logging.debug(f"start to receive image with {size} bytes")
        # 接收图像数据
        data = bytearray()  # 可变字节数据对象
        # 分多次接收图像数据
        while len(data) < size:
            packet = conn.recv(min(4096, size - len(data)))
            if not packet:
                raise RuntimeError("Connection closed during data transfer")
            data.extend(packet)  # 使用 extend 追加多个字节
        # 全部接收后检测实际接收长度是否和包头相同
        if len(data) < size:
            raise RuntimeError(
                f"Incomplete data, expected {size} bytes but only {len(data)} bytes"
            )
        elif len(data) > size:
            logging.warning(
                f"Received more data than expected, expected {size} bytes but actually received {len(data)} bytes"
            )
        return data

    def process_image(self, data):
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None or len(img) == 0:
            raise RuntimeError("Failed to decode image")
        return img

    def receive_worker(self, conn, running_flag):
        """接收并显示图片的线程函数，持续接收直到被停止"""
        t_sum = 0.0
        t_log = 0
        try:
            while running_flag[0]:
                t1 = time.perf_counter()
                img_byte = self.handle_connect(conn)

                # 诊断：检查时间差
                elapsed = time.perf_counter() - t1
                logging.debug(f"Received image, elapsed={elapsed:.9f}s")
                t_sum += elapsed
                t_log += 1
                if t_log % 10 == 0:
                    logging.info(f"receive thread frequence:{1 / (t_sum / 10):.6f}Hz")
                    t_sum = 0.0
                    t_log = 0

                img = self.process_image(img_byte)
                cv2.imshow("receive_image", img)
                if cv2.waitKey(1) & 0xFF == 27:
                    logging.info("interrupt by user.")
                    break

        except (ConnectionResetError, BrokenPipeError) as e:
            logging.error(f"Connection error: {e}")
        except RuntimeError as e:
            logging.error(f"Runtime error in receive_worker: {e}")
        except Exception as e:
            logging.error(f"Unexpected error in receive_worker: {e}")
        finally:
            cv2.destroyAllWindows()
            conn.close()


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 12345

    # 方式1: 使用上下文管理器（推荐）- 自动释放资源
    try:
        with ImageReceiver(host, port) as img_rec:
            img_rec.receive_picture()
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    except RuntimeError as e:
        logging.error(f"Runtime error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    # 退出 with 块时自动调用 close() 释放资源

    # 方式2: 手动管理 - 需要显式调用 close()
    # img_rec = ImageReceiver(host, port)
    # try:
    #     img_rec.receive_picture()
    # except Exception as e:
    #     logging.error(f"Error: {e}")
    # finally:
    #     img_rec.close()
