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
        image[202, 202] = bright_bgr
        image[230, 230] = bright_bgr
        image[20, 20] = green_bgr

        masks = build_masks(image, Config())

        self.assertEqual(255, int(masks.green[200, 200]))
        self.assertEqual(255, int(masks.red[210, 210]))
        self.assertEqual(255, int(masks.bright_raw[202, 202]))
        self.assertEqual(255, int(masks.bright[202, 202]))
        self.assertEqual(255, int(masks.bright_raw[230, 230]))
        self.assertEqual(0, int(masks.bright[230, 230]))
        self.assertEqual(255, int(masks.color_support[200, 200]))
        self.assertEqual(255, int(masks.color_support[210, 210]))
        self.assertEqual(13, masks.support_kernel_size)
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
    def test_color_support_kernel_scales_to_odd_sizes(self):
        self.assertEqual(3, detector._scaled_odd_length(15, 0.1))
        self.assertEqual(15, detector._scaled_odd_length(15, 1.0))
        self.assertEqual(31, detector._scaled_odd_length(15, 2.0))

    def test_high_resolution_keeps_minimum_thresholds_for_distant_light(self):
        bright = np.zeros((960, 1280), dtype=np.uint8)
        bright[200:213, 200:216] = 255
        green = np.zeros_like(bright)
        green[190:223, 190:226] = 255
        masks = detector.Masks(
            roi=np.full_like(bright, 255),
            green=green,
            red=np.zeros_like(bright),
            bright=bright,
            roi_rect=(0, 0, 1280, 960),
            scale=2.0,
        )

        candidate, diagnostics = detector._find_candidate_with_diagnostics(
            masks
        )

        self.assertIsNotNone(candidate)
        self.assertEqual((200, 200, 16, 13), candidate.bbox)
        self.assertEqual(15, diagnostics.min_width)
        self.assertEqual(12, diagnostics.min_height)
        self.assertEqual(80, diagnostics.min_area)
        self.assertEqual(200, diagnostics.min_color_score)
        self.assertEqual(180, diagnostics.max_width)
        self.assertEqual(180, diagnostics.max_height)

    def test_sparse_bright_component_is_rejected_by_fill_density(self):
        bright = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(bright, (200, 200), (289, 244), 255, thickness=1)
        green = np.zeros_like(bright)
        green[190:255, 190:300] = 255
        masks = detector.Masks(
            roi=np.full_like(bright, 255),
            green=green,
            red=np.zeros_like(bright),
            bright=bright,
            roi_rect=(0, 0, 640, 480),
            scale=1.0,
        )

        candidate, diagnostics = detector._find_candidate_with_diagnostics(
            masks
        )

        self.assertIsNone(candidate)
        self.assertEqual(1, diagnostics.rejected_component_fill)

    def test_candidate_ranking_penalizes_sparse_component(self):
        bright = np.zeros((200, 400), dtype=np.uint8)
        cv2.rectangle(bright, (40, 50), (79, 79), 255, thickness=2)
        cv2.rectangle(bright, (200, 50), (219, 69), 255, thickness=-1)
        green = np.zeros_like(bright)
        green[40:90, 30:90] = 255
        green[44:76, 194:226] = 255
        masks = detector.Masks(
            roi=np.full_like(bright, 255),
            green=green,
            red=np.zeros_like(bright),
            bright=bright,
            roi_rect=(0, 0, 400, 200),
            scale=1.0,
        )

        candidate, diagnostics = detector._find_candidate_with_diagnostics(
            masks
        )

        self.assertEqual((200, 50, 20, 20), candidate.bbox)
        self.assertEqual(2, diagnostics.accepted_candidates)
        first, second = diagnostics.top_candidates
        self.assertGreater(second.color_score, first.color_score)
        self.assertGreater(first.selection_score, second.selection_score)

    def test_arrow_like_component_has_fill_density_margin(self):
        mask = make_arrow_mask("right")
        area = cv2.countNonZero(mask)
        fill_density = area / mask.size

        self.assertGreater(
            fill_density, detector.DEFAULT_CONFIG.min_component_fill_density
        )

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


def make_mask_from_logged_band_counts(counts):
    height = 19
    mask = np.zeros((height, 20), dtype=np.uint8)
    band_ranges = (
        (0, 3),
        (3, 6),
        (6, 9),
        (9, 12),
        (12, 14),
        (14, 16),
        (16, 18),
        (18, 20),
    )
    for (start, end), count in zip(band_ranges, counts):
        band_width = end - start
        base, extra = divmod(count, band_width)
        for offset, column in enumerate(range(start, end)):
            pixels = base + (1 if offset < extra else 0)
            top = (height - pixels) // 2
            mask[top : top + pixels, column] = 255
    return mask


class ArrowClassificationTests(unittest.TestCase):
    def test_available_picture_regression_set_keeps_expected_labels(self):
        expected_labels = {
            "00010.jpg": "right",
            "00017.jpg": "left",
            "00107.jpg": "right",
            "001_0033.jpg": "unknown",
            "00299.jpg": "unknown",
            "003_0030.jpg": "left",
            "004_0001.jpg": "stop",
            "006_0018.jpg": "left",
            "01042.jpg": "straight",
            "capture_1779365603.jpg": "right",
            "capture_1779365606.jpg": "right",
            "capture_1779365607.jpg": "right",
            "capture_1779365615.jpg": "right",
        }

        for filename, expected in expected_labels.items():
            with self.subTest(filename=filename):
                image = cv2.imread(str(PICTURES_DIR / filename))
                self.assertIsNotNone(image)
                self.assertEqual(expected, detect_traffic_light(image).label)

    def test_color_support_recovers_arrows_from_bright_background(self):
        expected_labels = {
            "capture_1786544471347320238_000720.jpg": "right",
            "capture_1786544476988825876_000839.jpg": "left",
            "capture_1786544480705245656_000920.jpg": "straight",
        }

        for filename, expected in expected_labels.items():
            with self.subTest(filename=filename):
                image = cv2.imread(str(PICTURES_DIR / filename))
                self.assertIsNotNone(image)
                detection = detect_traffic_light(image)

                self.assertEqual(expected, detection.label)
                self.assertIsNotNone(detection.bbox)
                self.assertGreater(detection.selection_score, 0.0)

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

    def test_unequal_band_widths_do_not_bias_right_arrow_to_left(self):
        logged_right_bands = (
            (15, 17, 16, 38, 36, 29, 21, 13),
            (12, 15, 16, 37, 37, 29, 21, 12),
            (13, 15, 16, 39, 36, 29, 20, 13),
        )

        for counts in logged_right_bands:
            with self.subTest(counts=counts):
                result = classify_green_shape(
                    make_mask_from_logged_band_counts(counts)
                )

                self.assertEqual("right", result.label)
                self.assertEqual(4, result.projection_peak)
                self.assertGreater(
                    result.diagnostics.projection_density[4],
                    result.diagnostics.projection_density[3],
                )


class DrawingAndFixedDemoTests(unittest.TestCase):
    def test_save_camera_frame_writes_to_requested_pictures_directory(self):
        image = cv2.imread(str(PICTURES_DIR / "004_0001.jpg"))
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = detector.save_camera_frame(
                image,
                frame_id=7,
                output_dir=Path(output_dir),
                timestamp_ns=123456789,
            )

            self.assertEqual(
                "capture_123456789_000007.jpg", output_path.name
            )
            self.assertEqual(Path(output_dir).resolve(), output_path.parent)
            saved = cv2.imread(str(output_path))
            self.assertIsNotNone(saved)
            self.assertEqual(image.shape, saved.shape)

    def test_space_key_saves_current_clean_camera_frame(self):
        image = cv2.imread(str(PICTURES_DIR / "004_0001.jpg"))
        camera = mock.Mock()
        camera.get_picture.return_value = image
        camera.correct_img.return_value = image

        with mock.patch(
            "camera_capture.CameraCapture", return_value=camera
        ), mock.patch.object(detector.cv2, "namedWindow"), mock.patch.object(
            detector.cv2, "imshow"
        ), mock.patch.object(
            detector.cv2, "waitKey", side_effect=(ord(" "), ord("q"))
        ), mock.patch.object(
            detector.cv2, "destroyAllWindows"
        ), mock.patch.object(
            detector, "save_camera_frame", return_value=Path("capture.jpg")
        ) as save:
            status = detector.run_camera()

        self.assertEqual(0, status)
        save.assert_called_once()
        self.assertEqual(1, save.call_args.args[1])
        camera.close.assert_called_once_with()

    def test_tuning_log_contains_masks_filters_and_selected_candidate_scores(self):
        image = cv2.imread(str(PICTURES_DIR / "004_0001.jpg"))
        detection, diagnostics = detector.analyze_traffic_light(image)

        message = detector.format_detection_log(
            12, detection, diagnostics, fps=29.5, detection_ms=3.2
        )

        for field_name in (
            "label=stop",
            "green_score=",
            "red_score=",
            "color_density=",
            "component_fill=",
            "selection_score=",
            "support_kernel=",
            "bright_raw:",
            "color_support:",
            "bright_supported:",
            "mask_pixels=",
            "limits=",
            "components=",
            "rejected=",
        ):
            self.assertIn(field_name, message)

        candidate_message = detector.format_candidate_debug(
            diagnostics.search.top_candidates
        )
        self.assertIn("top_candidates=[#1(", candidate_message)
        self.assertNotIn("#4(", candidate_message)

    def test_realtime_log_tracker_throttles_noisy_state_changes(self):
        tracker = detector.RealtimeLogTracker(interval_seconds=1.0)
        unknown = Detection(label="unknown")
        stop = Detection(label="stop", color="red")

        self.assertEqual(("initial", 0), tracker.update(unknown, now=0.0))
        self.assertIsNone(tracker.update(stop, now=0.1))
        self.assertIsNone(tracker.update(unknown, now=0.2))
        self.assertEqual(("periodic", 2), tracker.update(unknown, now=1.0))

    def test_camera_stream_always_corrects_each_frame(self):
        image = cv2.imread(str(PICTURES_DIR / "004_0001.jpg"))
        camera = mock.Mock()
        camera.get_picture.return_value = image
        camera.correct_img.return_value = image

        with mock.patch(
            "camera_capture.CameraCapture", return_value=camera
        ), mock.patch.object(detector.cv2, "namedWindow"), mock.patch.object(
            detector.cv2, "imshow"
        ), mock.patch.object(
            detector.cv2, "waitKey", return_value=ord("q")
        ), mock.patch.object(
            detector.cv2, "destroyAllWindows"
        ):
            status = detector.run_camera()

        self.assertEqual(0, status)
        camera.get_picture.assert_called_once_with()
        camera.correct_img.assert_called_once()
        camera.close.assert_called_once_with()

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

    def test_main_uses_camera_stream_by_default(self):
        with mock.patch.object(detector, "run_camera", return_value=0) as camera:
            status = detector.main([])

        self.assertEqual(0, status)
        camera.assert_called_once_with(
            camera_index=0,
            width=640,
            height=480,
            log_interval=1.0,
        )


if __name__ == "__main__":
    unittest.main()
