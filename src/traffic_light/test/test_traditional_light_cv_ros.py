import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PACKAGE_DIR / "scripts"
PICTURES_DIR = PACKAGE_DIR / "pictures"
sys.path.insert(0, str(SCRIPT_DIR))

from traditional_light_cv_ros import (  # noqa: E402
    IMAGE_TOPIC,
    DetectionLogTracker,
    FpsMeter,
    annotate_frame,
    draw_fps,
    format_detection_log,
)


class FpsAndOverlayTests(unittest.TestCase):
    def test_fps_meter_uses_elapsed_seconds(self):
        meter = FpsMeter()

        self.assertEqual(0.0, meter.update(now=10.0))
        self.assertAlmostEqual(10.0, meter.update(now=10.1), places=6)

    def test_draw_fps_changes_only_upper_right_overlay_region(self):
        image = np.zeros((120, 320, 3), dtype=np.uint8)

        result = draw_fps(image.copy(), 12.5)

        changed = np.any(result != image, axis=2)
        self.assertTrue(changed.any())
        ys, xs = np.where(changed)
        self.assertGreater(xs.min(), image.shape[1] // 2)
        self.assertLessEqual(ys.max(), 35)

    def test_annotate_frame_reuses_traditional_detector(self):
        image = cv2.imread(str(PICTURES_DIR / "02051.jpg"))
        meter = FpsMeter()

        annotated, detection, fps = annotate_frame(image, meter, now=1.0)

        self.assertEqual("right", detection.label)
        self.assertEqual(0.0, fps)
        self.assertFalse(np.array_equal(image, annotated))

    def test_image_topic_is_fixed_to_requested_camera_topic(self):
        self.assertEqual("/ucar_camera/image_raw", IMAGE_TOPIC)

    def test_detection_log_contains_direction_diagnostics(self):
        image = cv2.imread(str(PICTURES_DIR / "02051.jpg"))
        _, detection, _ = annotate_frame(image, FpsMeter(), now=1.0)

        message = format_detection_log(detection)

        self.assertIn("bbox_ratio=", message)
        self.assertIn("pca_abs=", message)
        self.assertIn("axis_margin=", message)
        self.assertIn("eig_ratio=", message)
        self.assertIn("bands=", message)
        self.assertIn("density=", message)

    def test_log_tracker_reports_classification_state_changes_immediately(self):
        tracker = DetectionLogTracker()
        image = cv2.imread(str(PICTURES_DIR / "003_0030.jpg"))
        _, left, _ = annotate_frame(image, FpsMeter(), now=1.0)
        straight = left.__class__(
            label="straight",
            bbox=left.bbox,
            color=left.color,
            color_score=left.color_score,
            component_area=left.component_area,
            orientation="vertical",
        )

        self.assertTrue(tracker.update(left))
        self.assertFalse(tracker.update(left))
        self.assertTrue(tracker.update(straight))


if __name__ == "__main__":
    unittest.main()
