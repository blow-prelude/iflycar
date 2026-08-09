"""ROS image-stream wrapper for the traditional traffic-light detector."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from traditional_light_cv import Detection, detect_traffic_light, draw_detection

try:
    import rospy
    from cv_bridge import CvBridge, CvBridgeError
    from sensor_msgs.msg import Image
except ImportError:  # Keep FPS and frame-processing helpers testable off-ROS.
    rospy = None
    CvBridge = None
    CvBridgeError = Exception
    Image = None


IMAGE_TOPIC = "/ucar_camera/image_raw"
WINDOW_NAME = "traditional_light_cv_ros"
DISPLAY_RATE_HZ = 30.0


class FpsMeter:
    """Smoothed FPS calculator driven by a monotonic timestamp."""

    def __init__(self, smoothing: float = 0.2) -> None:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        self._smoothing = smoothing
        self._previous_time: float | None = None
        self._fps = 0.0

    def update(self, now: float | None = None) -> float:
        timestamp = time.perf_counter() if now is None else float(now)
        if self._previous_time is not None:
            elapsed = timestamp - self._previous_time
            if elapsed > 0.0:
                instantaneous = 1.0 / elapsed
                if self._fps == 0.0:
                    self._fps = instantaneous
                else:
                    self._fps = (
                        self._smoothing * instantaneous
                        + (1.0 - self._smoothing) * self._fps
                    )
        self._previous_time = timestamp
        return self._fps


def draw_fps(image: np.ndarray, fps: float) -> np.ndarray:
    """Draw only the FPS text in the image's upper-right corner."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a three-channel BGR image")

    text = f"FPS: {max(0.0, float(fps)):.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    margin = 10
    origin = (
        max(margin, image.shape[1] - text_width - margin),
        margin + text_height,
    )
    cv2.putText(
        image,
        text,
        origin,
        font,
        scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return image


def annotate_frame(
    frame: np.ndarray,
    fps_meter: FpsMeter,
    now: float | None = None,
) -> tuple[np.ndarray, Detection, float]:
    """Run the shared detector, draw its result, then overlay the FPS."""
    detection = detect_traffic_light(frame)
    canvas = draw_detection(frame, detection)
    fps = fps_meter.update(now=now)
    draw_fps(canvas, fps)
    return canvas, detection, fps


class TrafficLightRosNode:
    """Subscribe to the camera image topic and display annotated frames."""

    def __init__(self) -> None:
        if rospy is None or CvBridge is None or Image is None:
            raise RuntimeError(
                "ROS dependencies are unavailable; run this node in a ROS environment"
            )

        self._bridge = CvBridge()
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._fps_meter = FpsMeter()
        self._subscriber = rospy.Subscriber(
            IMAGE_TOPIC,
            Image,
            self._image_callback,
            queue_size=1,
            buff_size=2**24,
        )
        rospy.loginfo("Subscribed to %s", IMAGE_TOPIC)

    def _image_callback(self, message: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except CvBridgeError as error:
            rospy.logerr("CvBridge conversion failed: %s", error)
            return

        with self._frame_lock:
            self._latest_frame = frame.copy()

    def _latest(self) -> np.ndarray | None:
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def run(self) -> None:
        rate = rospy.Rate(DISPLAY_RATE_HZ)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        try:
            while not rospy.is_shutdown():
                frame = self._latest()
                if frame is None:
                    rate.sleep()
                    continue

                canvas, detection, _ = annotate_frame(frame, self._fps_meter)
                rospy.loginfo_throttle(
                    1.0,
                    "label=%s bbox=%s color=%s score=%d area=%d axis=%s peak=%s",
                    detection.label,
                    detection.bbox,
                    detection.color,
                    detection.color_score,
                    detection.component_area,
                    detection.orientation,
                    detection.projection_peak,
                )
                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    rospy.signal_shutdown("display window closed")
                rate.sleep()
        finally:
            cv2.destroyWindow(WINDOW_NAME)


def main() -> None:
    if rospy is None:
        raise RuntimeError(
            "ROS dependencies are unavailable; run this node in a ROS environment"
        )
    rospy.init_node("traditional_light_cv_ros", anonymous=False)
    TrafficLightRosNode().run()


if __name__ == "__main__":
    main()
