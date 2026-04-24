"""test_fill_missing_line.py — _fill_missing_line 单元测试"""
import os
import numpy as np
import pytest
from image_process import ImageProcess


class TestFillMissingLineLeftMissing:
    """左线缺失，右线存在的场景"""

    def setup_method(self):
        self.proc = ImageProcess()
        self.img_shape = (240, 320)

    def test_returns_true_when_left_missing(self):
        self.proc.right_line = [(160, 200), (162, 198), (164, 196), (166, 194)]
        result = self.proc._fill_missing_line(self.img_shape)
        assert result is True

    def test_supple_left_filled_with_boundary(self):
        self.proc.right_line = [(160, 200), (162, 198), (164, 196), (166, 194)]
        self.proc._fill_missing_line(self.img_shape)
        assert len(self.proc.supple_left_line) > 0
        for point in self.proc.supple_left_line:
            assert point[0] == 0

    def test_supple_right_filled_via_interpolation(self):
        self.proc.right_line = [(160, 200), (162, 198), (164, 196), (166, 194)]
        self.proc._fill_missing_line(self.img_shape)
        assert len(self.proc.supple_right_line) > 0

    def test_both_supple_lines_same_length(self):
        self.proc.right_line = [(160, 200), (162, 198), (164, 196), (166, 194)]
        self.proc._fill_missing_line(self.img_shape)
        assert len(self.proc.supple_left_line) == len(self.proc.supple_right_line)

    def test_y_coordinates_aligned(self):
        self.proc.right_line = [(160, 200), (162, 198), (164, 196), (166, 194)]
        self.proc._fill_missing_line(self.img_shape)
        for lp, rp in zip(self.proc.supple_left_line, self.proc.supple_right_line):
            assert lp[1] == rp[1]

    def test_bottom_fill_extends_to_img_bottom(self):
        self.proc.right_line = [(160, 200), (162, 198), (164, 196), (166, 194)]
        self.proc._fill_missing_line(self.img_shape)
        first_y = self.proc.supple_right_line[0][1]
        assert first_y >= 237


class TestFillMissingLineRightMissing:
    """右线缺失，左线存在的场景"""

    def setup_method(self):
        self.proc = ImageProcess()
        self.img_shape = (240, 320)

    def test_returns_true_when_right_missing(self):
        self.proc.left_line = [(160, 200), (158, 198), (156, 196), (154, 194)]
        result = self.proc._fill_missing_line(self.img_shape)
        assert result is True

    def test_supple_right_filled_with_boundary(self):
        self.proc.left_line = [(160, 200), (158, 198), (156, 196), (154, 194)]
        self.proc._fill_missing_line(self.img_shape)
        assert len(self.proc.supple_right_line) > 0
        for point in self.proc.supple_right_line:
            assert point[0] == self.img_shape[1] - 1

    def test_both_supple_lines_same_length(self):
        self.proc.left_line = [(160, 200), (158, 198), (156, 196), (154, 194)]
        self.proc._fill_missing_line(self.img_shape)
        assert len(self.proc.supple_left_line) == len(self.proc.supple_right_line)

    def test_y_coordinates_aligned(self):
        self.proc.left_line = [(160, 200), (158, 198), (156, 196), (154, 194)]
        self.proc._fill_missing_line(self.img_shape)
        for lp, rp in zip(self.proc.supple_left_line, self.proc.supple_right_line):
            assert lp[1] == rp[1]


class TestFillMissingLineEdgeCases:
    """边界情况"""

    def setup_method(self):
        self.proc = ImageProcess()
        self.img_shape = (240, 320)

    def test_returns_false_when_both_empty(self):
        assert self.proc._fill_missing_line(self.img_shape) is False

    def test_returns_false_when_both_present(self):
        self.proc.left_line = [(100, 200), (98, 198)]
        self.proc.right_line = [(200, 200), (202, 198)]
        assert self.proc._fill_missing_line(self.img_shape) is False

    def test_no_side_effects_when_both_present(self):
        self.proc.left_line = [(100, 200), (98, 198)]
        self.proc.right_line = [(200, 200), (202, 198)]
        self.proc._fill_missing_line(self.img_shape)
        assert len(self.proc.supple_left_line) == 0
        assert len(self.proc.supple_right_line) == 0

    def test_single_point_existing_line(self):
        self.proc.right_line = [(160, 200)]
        result = self.proc._fill_missing_line(self.img_shape)
        assert result is True
        assert len(self.proc.supple_left_line) == len(self.proc.supple_right_line)
