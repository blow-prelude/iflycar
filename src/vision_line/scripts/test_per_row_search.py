"""test_per_row_search.py — 按行递推搜索策略单元测试"""
import numpy as np
import pytest
from image_process import ImageProcess


def _make_binary_with_edges(h=240, w=320, left_x=145, right_x=175):
    """生成白底二值图，左、右各一条竖直黑边线（宽3px）

    默认 left_x/right_x 选在逐行递推搜索的初始窗口内：
    左侧搜索范围 [mid_x+30-range, mid_x+30]，右侧搜索范围 [mid_x-30, mid_x-30+range]。
    """
    img = np.full((h, w), 255, dtype=np.uint8)
    img[:, left_x - 1 : left_x + 2] = 0
    img[:, right_x - 1 : right_x + 2] = 0
    return img


class TestPerRowSearchBasic:
    """验证按行递推搜索基本行为"""

    def setup_method(self):
        self.proc = ImageProcess()

    def test_finds_both_edges(self):
        """应同时检测到左右边线"""
        binary = _make_binary_with_edges()
        canvas = np.zeros((240, 320, 3), dtype=np.uint8)
        self.proc.get_side_line_task_1(binary, canvas)
        assert len(self.proc.left_line) > 0
        assert len(self.proc.right_line) > 0

    def test_left_line_y_descending(self):
        """左线点 y 应降序排列（从底到顶）"""
        binary = _make_binary_with_edges()
        canvas = np.zeros((240, 320, 3), dtype=np.uint8)
        self.proc.get_side_line_task_1(binary, canvas)
        for i in range(len(self.proc.left_line) - 1):
            assert self.proc.left_line[i][1] > self.proc.left_line[i + 1][1]

    def test_right_line_y_descending(self):
        """右线点 y 应降序排列"""
        binary = _make_binary_with_edges()
        canvas = np.zeros((240, 320, 3), dtype=np.uint8)
        self.proc.get_side_line_task_1(binary, canvas)
        for i in range(len(self.proc.right_line) - 1):
            assert self.proc.right_line[i][1] > self.proc.right_line[i + 1][1]

    def test_midline_populated(self):
        """中线应被计算"""
        binary = _make_binary_with_edges()
        canvas = np.zeros((240, 320, 3), dtype=np.uint8)
        self.proc.get_side_line_task_1(binary, canvas)
        assert len(self.proc.mid_line) > 0

    def test_no_prev_frame_still_works(self):
        """无上一帧数据时也应正常工作（从 mid_x 起始）"""
        self.proc.prev_left_line = []
        self.proc.prev_right_line = []
        self.proc.prev_supple_left_line = []
        self.proc.prev_supple_right_line = []
        binary = _make_binary_with_edges()
        canvas = np.zeros((240, 320, 3), dtype=np.uint8)
        self.proc.get_side_line_task_1(binary, canvas)
        assert len(self.proc.left_line) > 0
        assert len(self.proc.right_line) > 0


class TestPerRowSearchRange:
    """验证搜索窗口动态缩小行为"""

    def setup_method(self):
        self.proc = ImageProcess()

    def test_narrow_range_still_finds_edge(self):
        """即使在上半部分（range=30），仍应检测到边线"""
        binary = _make_binary_with_edges(h=240, w=320, left_x=148, right_x=172)
        canvas = np.zeros((240, 320, 3), dtype=np.uint8)
        self.proc.get_side_line_task_1(binary, canvas)
        upper_points = [p for p in self.proc.left_line if p[1] < 144]
        assert len(upper_points) > 0


class TestPerRowSearchFallback:
    """验证 miss 回退行为"""

    def setup_method(self):
        self.proc = ImageProcess()

    def test_one_side_missing_still_produces_supple(self):
        """当一侧完全丢线时，_fill_missing_line 应补出边界线"""
        h, w = 240, 320
        img = np.full((h, w), 255, dtype=np.uint8)
        # 放在右侧搜索窗口内: [mid_x-30, mid_x-30+range] = [130, 180]
        img[:, 169:172] = 0
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        self.proc.get_side_line_task_1(img, canvas)
        assert len(self.proc.supple_left_line) > 0 or len(self.proc.supple_right_line) > 0
