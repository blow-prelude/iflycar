import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PACKAGE_DIR / "scripts"
PICTURES_DIR = PACKAGE_DIR / "pictures"
sys.path.insert(0, str(SCRIPT_DIR))

from traditional_light_cv import (
    Config,
    build_masks,
    classify_green_shape,
    detect_traffic_light,
)


class MaskAndValidationTests(unittest.TestCase):
    def test_build_masks_applies_hsv_rules_inside_roi_only(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        green_bgr = cv2.cvtColor(
            np.uint8([[[60, 200, 230]]]), cv2.COLOR_HSV2BGR
        )[0, 0]
        red_bgr = cv2.cvtColor(
            np.uint8([[[5, 200, 230]]]), cv2.COLOR_HSV2BGR
        )[0, 0]
        bright_bgr = cv2.cvtColor(
            np.uint8([[[90, 50, 240]]]), cv2.COLOR_HSV2BGR
        )[0, 0]

        image[200, 200] = green_bgr
        image[210, 210] = red_bgr
        image[220, 220] = bright_bgr
        image[20, 20] = green_bgr

        masks = build_masks(image, Config())

        self.assertEqual(255, int(masks.green[200, 200]))
        self.assertEqual(255, int(masks.red[210, 210]))
        self.assertEqual(255, int(masks.bright[220, 220]))
        self.assertEqual(0, int(masks.green[20, 20]))
        self.assertEqual((160, 134, 499, 374), masks.roi_rect)
        self.assertAlmostEqual(1.0, masks.scale)

    def test_build_masks_rejects_invalid_images(self):
        invalid_images = (
            None,
            np.array([], dtype=np.uint8),
            np.zeros((480, 640), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.float32),
        )

        for image in invalid_images:
            with self.subTest(shape=getattr(image, "shape", None)):
                with self.assertRaises(ValueError):
                    build_masks(image, Config())


class CandidateDetectionTests(unittest.TestCase):
    def test_blank_image_returns_unknown(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        detection = detect_traffic_light(image)

        self.assertEqual("unknown", detection.label)
        self.assertIsNone(detection.bbox)
        self.assertEqual("unknown", detection.color)

    def test_red_sample_returns_stop(self):
        image = cv2.imread(str(PICTURES_DIR / "004_0001.jpg"))

        detection = detect_traffic_light(image)

        self.assertEqual("stop", detection.label)
        self.assertEqual("red", detection.color)
        self.assertIsNotNone(detection.bbox)
        self.assertGreaterEqual(detection.color_score, 200)

    def test_stop_candidate_bbox_is_inside_normalized_roi(self):
        image = cv2.imread(str(PICTURES_DIR / "004_0001.jpg"))

        detection = detect_traffic_light(image)

        x, y, width, height = detection.bbox
        roi_x1, roi_y1, roi_x2, roi_y2 = build_masks(image).roi_rect
        self.assertGreaterEqual(x, roi_x1)
        self.assertGreaterEqual(y, roi_y1)
        self.assertLessEqual(x + width, roi_x2)
        self.assertLessEqual(y + height, roi_y2)


def make_arrow_mask(direction):
    right = np.zeros((48, 64), dtype=np.uint8)
    cv2.rectangle(right, (8, 20), (40, 28), 255, thickness=-1)
    cv2.fillConvexPoly(
        right,
        np.array([(36, 7), (59, 24), (36, 41)], dtype=np.int32),
        255,
    )
    if direction == "right":
        return right
    if direction == "left":
        return np.fliplr(right).copy()
    if direction == "straight":
        return cv2.rotate(right, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(direction)


class ArrowClassificationTests(unittest.TestCase):
    def test_synthetic_arrow_shapes_have_expected_directions(self):
        for expected in ("straight", "left", "right"):
            with self.subTest(expected=expected):
                result = classify_green_shape(make_arrow_mask(expected))
                self.assertEqual(expected, result.label)

    def test_four_supplied_images_have_expected_labels(self):
        expected_labels = {
            "02051.jpg": "right",
            "02052.jpg": "straight",
            "003_0030.jpg": "left",
            "004_0001.jpg": "stop",
        }

        for filename, expected in expected_labels.items():
            with self.subTest(filename=filename):
                image = cv2.imread(str(PICTURES_DIR / filename))
                self.assertIsNotNone(image)
                detection = detect_traffic_light(image)
                self.assertEqual(expected, detection.label)
                self.assertIsNotNone(detection.bbox)
                x, y, width, height = detection.bbox
                roi_x1, roi_y1, roi_x2, roi_y2 = build_masks(image).roi_rect
                self.assertGreaterEqual(x, roi_x1)
                self.assertGreaterEqual(y, roi_y1)
                self.assertLessEqual(x + width, roi_x2)
                self.assertLessEqual(y + height, roi_y2)


if __name__ == "__main__":
    unittest.main()
