import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


def load_image_process_module():
    """Load image_process without its camera/network runtime dependencies."""
    for module_name, class_name in (
        ("camera_capture", "CameraCapture"),
        ("picture_cli", "ImageSender"),
    ):
        module = types.ModuleType(module_name)
        setattr(module, class_name, type(class_name, (), {}))
        sys.modules[module_name] = module

    module_path = Path(__file__).resolve().parents[1] / "scripts" / "image_process.py"
    spec = importlib.util.spec_from_file_location(
        "vision_line_image_process_under_test",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load image_process module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


image_process = load_image_process_module()


class ImageProcessPreprocessTest(unittest.TestCase):
    @staticmethod
    def make_zero_padded_frame(with_bright_line=False):
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        frame[:, 30:130] = 100
        valid_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        valid_mask[:, 30:130] = 255
        if with_bright_line:
            frame[20:80, 78:82] = 240
        return frame, valid_mask

    def test_explicit_mask_prevents_white_edge_at_black_padding_boundary(self):
        frame, valid_mask = self.make_zero_padded_frame()

        result = image_process.ImageProcess().preprocess(
            frame,
            valid_mask=valid_mask,
        )

        self.assertEqual(0, np.count_nonzero(result[10:90, 28:35]))
        self.assertEqual(0, np.count_nonzero(result[10:90, 125:132]))

    def test_explicit_mask_preserves_bright_line_inside_valid_region(self):
        frame, valid_mask = self.make_zero_padded_frame(with_bright_line=True)

        result = image_process.ImageProcess().preprocess(
            frame,
            valid_mask=valid_mask,
        )

        self.assertGreater(np.count_nonzero(result[25:75, 78:82]), 0)

    def test_interpolated_mask_boundary_does_not_create_an_inner_white_edge(self):
        frame, valid_mask = self.make_zero_padded_frame()
        frame[:, 30] = 60
        frame[:, 129] = 60

        result = image_process.ImageProcess().preprocess(
            frame,
            valid_mask=valid_mask,
        )

        self.assertEqual(0, np.count_nonzero(result[10:90, 28:38]))
        self.assertEqual(0, np.count_nonzero(result[10:90, 122:132]))

    def test_zero_padding_is_inferred_when_mask_is_omitted(self):
        frame, _valid_mask = self.make_zero_padded_frame()

        result = image_process.ImageProcess().preprocess(frame)

        self.assertEqual(0, np.count_nonzero(result[10:90, 28:35]))
        self.assertEqual(0, np.count_nonzero(result[10:90, 125:132]))

    def test_mismatched_explicit_mask_is_rejected(self):
        frame, _valid_mask = self.make_zero_padded_frame()
        mismatched_mask = np.ones((99, 160), dtype=np.uint8)

        with self.assertRaisesRegex(RuntimeError, "valid_mask shape"):
            image_process.ImageProcess().preprocess(
                frame,
                valid_mask=mismatched_mask,
            )

    def test_empty_explicit_mask_is_rejected(self):
        frame, _valid_mask = self.make_zero_padded_frame()
        empty_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        with self.assertRaisesRegex(RuntimeError, "valid_mask must contain"):
            image_process.ImageProcess().preprocess(
                frame,
                valid_mask=empty_mask,
            )


if __name__ == "__main__":
    unittest.main()
