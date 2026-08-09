import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PACKAGE_DIR / "scripts"
PICTURES_DIR = PACKAGE_DIR / "pictures"
sys.path.insert(0, str(SCRIPT_DIR))

import traditional_light_cv as detector

from traditional_light_cv import (
    Config,
    Detection,
    DirectionDiagnostics,
    build_masks,
    classify_green_shape,
    detect_traffic_light,
    draw_detection,
    process_images,
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

    def test_direction_diagnostics_explain_pca_axis_decision(self):
        left = classify_green_shape(make_arrow_mask("left"))
        straight = classify_green_shape(make_arrow_mask("straight"))

        self.assertIsNotNone(left.diagnostics)
        self.assertGreater(left.diagnostics.axis_margin, 0.0)
        self.assertGreater(left.diagnostics.eigenvalue_ratio, 1.0)
        self.assertEqual(8, len(left.diagnostics.projection_bands))

        self.assertIsNotNone(straight.diagnostics)
        self.assertLess(straight.diagnostics.axis_margin, 0.0)
        self.assertGreater(straight.diagnostics.eigenvalue_ratio, 1.0)
        self.assertEqual(8, len(straight.diagnostics.projection_bands))

    def test_diagnostics_do_not_change_detection_equality(self):
        diagnostics = DirectionDiagnostics(
            principal_axis_abs=(0.9, 0.1),
            axis_margin=0.8,
            eigenvalue_ratio=2.0,
            projection_bands=(1, 2, 3, 4, 5, 6, 7, 8),
        )

        self.assertEqual(
            Detection(label="left"),
            Detection(label="left", diagnostics=diagnostics),
        )


class DrawingAndFixedDemoTests(unittest.TestCase):
    def test_draw_detection_only_draws_candidate_box_and_final_label(self):
        image = cv2.imread(str(PICTURES_DIR / "02051.jpg"))
        original = image.copy()
        detection = detect_traffic_light(image)

        with mock.patch.object(
            detector.cv2, "rectangle", wraps=cv2.rectangle
        ) as rectangle_mock, mock.patch.object(
            detector.cv2, "putText", wraps=cv2.putText
        ) as text_mock:
            annotated = draw_detection(image, detection)

        self.assertTrue(np.array_equal(original, image))
        self.assertFalse(np.array_equal(original, annotated))
        self.assertEqual(1, rectangle_mock.call_count)
        self.assertEqual(1, text_mock.call_count)
        self.assertEqual(detection.label, text_mock.call_args.args[1])

    def test_process_images_writes_four_annotations_and_terminal_metrics(self):
        with tempfile.TemporaryDirectory() as output_dir:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = process_images(
                    detector.SAMPLE_IMAGE_PATHS, Path(output_dir)
                )

            self.assertEqual(0, status)
            terminal_output = stdout.getvalue()
            expected_labels = {
                "02051.jpg": "right",
                "02052.jpg": "straight",
                "003_0030.jpg": "left",
                "004_0001.jpg": "stop",
            }
            for name, label in expected_labels.items():
                self.assertIn(f"{name}: label={label}", terminal_output)
                output_name = f"{Path(name).stem}_traditional{Path(name).suffix}"
                output_path = Path(output_dir) / output_name
                self.assertTrue(output_path.is_file(), output_path)
                self.assertGreater(output_path.stat().st_size, 0)
            self.assertIn("score=", terminal_output)
            self.assertIn("area=", terminal_output)
            self.assertIn("axis=", terminal_output)
            self.assertIn("peak=", terminal_output)
            self.assertNotIn("roi", terminal_output.lower())

    def test_process_images_continues_after_bad_input_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as output_dir:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                status = process_images(
                    (
                        PICTURES_DIR / "missing.jpg",
                        PICTURES_DIR / "004_0001.jpg",
                    ),
                    Path(output_dir),
                )

            self.assertEqual(1, status)
            self.assertIn("missing.jpg", stderr.getvalue())
            self.assertTrue(
                (Path(output_dir) / "004_0001_traditional.jpg").is_file()
            )

    def test_main_uses_code_defined_paths_without_parameters(self):
        with mock.patch.object(detector, "process_images", return_value=0) as batch:
            status = detector.main()

        self.assertEqual(0, status)
        batch.assert_called_once_with(
            detector.SAMPLE_IMAGE_PATHS, detector.OUTPUT_DIR
        )


if __name__ == "__main__":
    unittest.main()
