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
        assert first_y == 239  # 底部填充从 img_h - 1 = 239 开始


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


class TestIntegrationWithRealImage:
    """使用真实图片验证完整流程"""

    def _get_test_image(self):
        """加载测试图片并预处理"""
        pictures_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.pardir, "pictures"
        )
        test_img = os.path.join(pictures_dir, "test1.png")
        if not os.path.exists(test_img):
            pytest.skip("No test image available")
        proc = ImageProcess(img_path=test_img)
        binary = proc.preprocess()
        return proc, binary

    def test_full_pipeline_no_crash(self):
        """完整流程不应崩溃"""
        proc, binary = self._get_test_image()
        canvas = proc.return_frame()
        proc.get_side_line_task_1(binary, canvas, is_draw=False)

    def test_midline_via_manual_missing_line(self):
        """手动构造缺失线场景，验证中线计算正确"""
        proc, binary = self._get_test_image()
        img_shape = binary.shape
        proc3 = ImageProcess()
        proc3.right_line = [(160, 200), (162, 198), (164, 196)]
        proc3.left_line = []
        result = proc3._fill_missing_line(img_shape)
        assert result is True
        assert len(proc3.supple_left_line) > 0
        assert len(proc3.supple_right_line) > 0
        assert len(proc3.supple_left_line) == len(proc3.supple_right_line)
        for lp, rp in zip(proc3.supple_left_line, proc3.supple_right_line):
            assert lp[1] == rp[1]


class TestPointOrdering:
    """验证点的排列顺序"""

    def setup_method(self):
        self.proc = ImageProcess()
        self.img_shape = (240, 320)

    def test_descending_y_order_left_missing(self):
        """左线缺失时，supple 线应按 y 降序排列（从底部到顶部）"""
        self.proc.right_line = [(160, 200), (162, 198), (164, 196), (166, 194)]
        self.proc._fill_missing_line(self.img_shape)
        for line in [self.proc.supple_left_line, self.proc.supple_right_line]:
            for i in range(len(line) - 1):
                assert line[i][1] >= line[i + 1][1]

    def test_descending_y_order_right_missing(self):
        """右线缺失时，supple 线应按 y 降序排列"""
        self.proc.left_line = [(160, 200), (158, 198), (156, 196), (154, 194)]
        self.proc._fill_missing_line(self.img_shape)
        for line in [self.proc.supple_left_line, self.proc.supple_right_line]:
            for i in range(len(line) - 1):
                assert line[i][1] >= line[i + 1][1]

    def test_no_bottom_fill_when_line_at_bottom(self):
        """当已存在线的最低点已在图像底部时，不需要底部填充"""
        # bottom_y = 239 == img_h - 1, 所以 if bottom_y < img_h - 1 为 False
        self.proc.right_line = [(160, 239), (162, 237), (164, 235)]
        self.proc._fill_missing_line(self.img_shape)
        # 第一个点的 y 应该就是原始的 239，不会被额外填充
        assert self.proc.supple_right_line[0][1] == 239
        assert len(self.proc.supple_left_line) == len(self.proc.supple_right_line)
