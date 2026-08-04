# 缺失边线边界填充 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `get_side_line_task_1()` 中，当左线或右线缺失时，用图像边界替代缺失边线，使中线计算仍能正常进行。

**Architecture:** 新增 `_fill_missing_line()` 私有方法处理缺失边线逻辑，在搜索循环之后、插值之前调用。已存在的边线经过插值和底部填充后与边界线 y 对齐。

**Tech Stack:** Python 3.9, OpenCV (cv2), NumPy

**测试环境:** `D:\Anaconda\envs\opencv39\python.exe`，所有 pytest 命令需 `cd src/vision_line/scripts/` 后执行

**设计文档:** `docs/superpowers/specs/2026-04-24-missing-line-boundary-fill-design.md`

**注意事项:**
- 边界线生成使用已存在线的完整 y 列表（而非固定步长 -2），确保两边 y 坐标严格一一对应
- `_update_prev_frame_lines()` 的守卫条件 `if len(self.left_line) > 0 and len(self.right_line) > 0` 在缺失线场景下为 False，因此 prev_* 不会被更新。下一帧将回退到中线搜索。这是预期行为。

---

## 文件结构

| 操作 | 文件路径 | 职责 |
|------|---------|------|
| 修改 | `src/vision_line/scripts/image_process.py` | 新增 `_fill_missing_line()` 方法，修改 `get_side_line_task_1()` |
| 创建 | `src/vision_line/scripts/test_fill_missing_line.py` | 单元测试 |

---

### Task 1: 编写失败测试 — 左线缺失

**Files:**
- Create: `src/vision_line/scripts/test_fill_missing_line.py`
- Read: `src/vision_line/scripts/image_process.py:37-73` (构造函数参数), `332-404` (_linear_interpolation, _fill_boundary)

- [ ] **Step 1: 编写测试 — 左线缺失时右线保持、左线用 x=0 填充**

```python
"""test_fill_missing_line.py — _fill_missing_line 单元测试"""
import os
import numpy as np
import pytest
from image_process import ImageProcess


class TestFillMissingLineLeftMissing:
    """左线缺失，右线存在的场景"""

    def setup_method(self):
        self.proc = ImageProcess()
        # 模拟一张 240x320 的图像
        self.img_shape = (240, 320)

    def test_returns_true_when_left_missing(self):
        """左线为空、右线非空时返回 True"""
        self.proc.right_line = [(160, 200), (162, 198), (164, 196), (166, 194)]
        result = self.proc._fill_missing_line(self.img_shape)
        assert result is True

    def test_supple_left_filled_with_boundary(self):
        """supple_left_line 应该用 x=0 填充"""
        self.proc.right_line = [(160, 200), (162, 198), (164, 196), (166, 194)]
        self.proc._fill_missing_line(self.img_shape)
        assert len(self.proc.supple_left_line) > 0
        for point in self.proc.supple_left_line:
            assert point[0] == 0

    def test_supple_right_filled_via_interpolation(self):
        """supple_right_line 应该经过插值处理"""
        self.proc.right_line = [(160, 200), (162, 198), (164, 196), (166, 194)]
        self.proc._fill_missing_line(self.img_shape)
        assert len(self.proc.supple_right_line) > 0

    def test_both_supple_lines_same_length(self):
        """两条 supple 线长度必须一致"""
        self.proc.right_line = [(160, 200), (162, 198), (164, 196), (166, 194)]
        self.proc._fill_missing_line(self.img_shape)
        assert len(self.proc.supple_left_line) == len(self.proc.supple_right_line)

    def test_y_coordinates_aligned(self):
        """两条线的 y 坐标一一对应"""
        self.proc.right_line = [(160, 200), (162, 198), (164, 196), (166, 194)]
        self.proc._fill_missing_line(self.img_shape)
        for lp, rp in zip(self.proc.supple_left_line, self.proc.supple_right_line):
            assert lp[1] == rp[1]

    def test_bottom_fill_extends_to_img_bottom(self):
        """底部填充应将 supple_right 延伸到接近 img_h - 1"""
        self.proc.right_line = [(160, 200), (162, 198), (164, 196), (166, 194)]
        self.proc._fill_missing_line(self.img_shape)
        # 底部填充后第一个点的 y 应接近 img_h - 1
        first_y = self.proc.supple_right_line[0][1]
        assert first_y >= 237  # img_h - 1 = 239，步长 2 所以至少到 237 或 239
```

- [ ] **Step 2: 运行测试确认失败**

Run: (cd `src/vision_line/scripts/` then) `D:\Anaconda\envs\opencv39\python.exe -m pytest test_fill_missing_line.py -v`
Expected: FAIL — `AttributeError: 'ImageProcess' object has no attribute '_fill_missing_line'`

---

### Task 2: 编写失败测试 — 右线缺失 + 边界情况

**Files:**
- Modify: `src/vision_line/scripts/test_fill_missing_line.py`

- [ ] **Step 1: 添加右线缺失、两条线都空、两条线都存在、单点线的测试**

在 `test_fill_missing_line.py` 末尾追加：

```python
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
        """supple_right_line 应该用 x=img_w-1 填充"""
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
        """两条线都为空返回 False"""
        assert self.proc._fill_missing_line(self.img_shape) is False

    def test_returns_false_when_both_present(self):
        """两条线都存在返回 False"""
        self.proc.left_line = [(100, 200), (98, 198)]
        self.proc.right_line = [(200, 200), (202, 198)]
        assert self.proc._fill_missing_line(self.img_shape) is False

    def test_no_side_effects_when_both_present(self):
        """两条线都存在时不应修改 supple_*_line"""
        self.proc.left_line = [(100, 200), (98, 198)]
        self.proc.right_line = [(200, 200), (202, 198)]
        self.proc._fill_missing_line(self.img_shape)
        assert len(self.proc.supple_left_line) == 0
        assert len(self.proc.supple_right_line) == 0

    def test_single_point_existing_line(self):
        """存在线只有一个点时仍能正常处理"""
        self.proc.right_line = [(160, 200)]
        result = self.proc._fill_missing_line(self.img_shape)
        assert result is True
        assert len(self.proc.supple_left_line) == len(self.proc.supple_right_line)
```

- [ ] **Step 2: 运行所有测试确认失败**

Run: (cd `src/vision_line/scripts/` then) `D:\Anaconda\envs\opencv39\python.exe -m pytest test_fill_missing_line.py -v`
Expected: FAIL — `AttributeError`

---

### Task 3: 实现 `_fill_missing_line()` 方法

**Files:**
- Modify: `src/vision_line/scripts/image_process.py` — 在 `_fill_boundary` 方法的 `return left_line, right_line` 之后插入

- [ ] **Step 1: 在 `_fill_boundary` 方法之后、`fit_polynomial` 方法之前插入新方法**

在 `image_process.py` 的 `_fill_boundary` 方法末尾 (`return left_line, right_line` 之后) 插入：

```python
    def _fill_missing_line(self, img_shape):
        """当左线或右线缺失时，用图像边界替代缺失边线

        已存在的那条线会经过插值和底部边界填充，确保与边界线 y 对齐。
        边界线使用已存在线的完整 y 列表（而非固定步长），保证一一对应。

        Args:
            img_shape: 图像形状 (height, width, ...)

        Returns:
            bool: 是否填充了缺失线
        """
        img_h, img_w = img_shape[0], img_shape[1]

        # 左线缺失，右线存在
        if len(self.left_line) == 0 and len(self.right_line) > 0:
            supple_right = self._linear_interpolation(self.right_line)
            # 底部填充：从 img_h-1 延伸到右线最低点
            bottom_y = supple_right[0][1]
            if bottom_y < img_h - 1:
                ys = np.arange(img_h - 1, bottom_y, -2)
                xs = np.full_like(ys, img_w - 1)
                bottom_pts = list(zip(xs.tolist(), ys.tolist()))
                supple_right = bottom_pts + supple_right
            # 按右线的 y 列表生成左边界点，保证一一对应
            ys_all = [p[1] for p in supple_right]
            boundary_left = [(0, int(y)) for y in ys_all]
            self.supple_left_line = boundary_left
            self.supple_right_line = supple_right
            return True

        # 右线缺失，左线存在
        if len(self.right_line) == 0 and len(self.left_line) > 0:
            supple_left = self._linear_interpolation(self.left_line)
            # 底部填充
            bottom_y = supple_left[0][1]
            if bottom_y < img_h - 1:
                ys = np.arange(img_h - 1, bottom_y, -2)
                xs = np.zeros_like(ys)
                bottom_pts = list(zip(xs.tolist(), ys.tolist()))
                supple_left = bottom_pts + supple_left
            # 按左线的 y 列表生成右边界点
            ys_all = [p[1] for p in supple_left]
            boundary_right = [(img_w - 1, int(y)) for y in ys_all]
            self.supple_right_line = boundary_right
            self.supple_left_line = supple_left
            return True

        return False
```

- [ ] **Step 2: 运行所有测试确认通过**

Run: (cd `src/vision_line/scripts/` then) `D:\Anaconda\envs\opencv39\python.exe -m pytest test_fill_missing_line.py -v`
Expected: 全部 PASS

- [ ] **Step 3: 提交**

```bash
git add src/vision_line/scripts/image_process.py src/vision_line/scripts/test_fill_missing_line.py
git commit -m "feat: add _fill_missing_line() method with tests"
```

---

### Task 4: 集成到 `get_side_line_task_1()`

**Files:**
- Modify: `src/vision_line/scripts/image_process.py`
- 定位方式：搜索 `if len(self.left_line) > 0 and len(self.right_line) > 0:`，该代码块包含插值和 `_fill_boundary` 调用

- [ ] **Step 1: 替换原有插值代码块**

在 `get_side_line_task_1()` 方法中，找到以下代码块：

```python
            if len(self.left_line) > 0 and len(self.right_line) > 0:
                # 线性插值
                self.supple_left_line = self._linear_interpolation(self.left_line)
                self.supple_right_line = self._linear_interpolation(self.right_line)

                # 填充边界，使线段一直延伸到左右下角
                self.supple_left_line, self.supple_right_line = self._fill_boundary(
                    self.supple_left_line, self.supple_right_line, img.shape
                )
```

替换为：

```python
            filled = self._fill_missing_line(img.shape)
            if not filled:
                if len(self.left_line) > 0 and len(self.right_line) > 0:
                    # 线性插值
                    self.supple_left_line = self._linear_interpolation(self.left_line)
                    self.supple_right_line = self._linear_interpolation(self.right_line)

                    # 填充边界，使线段一直延伸到左右下角
                    self.supple_left_line, self.supple_right_line = self._fill_boundary(
                        self.supple_left_line, self.supple_right_line, img.shape
                    )
```

- [ ] **Step 2: 运行单元测试确认通过**

Run: (cd `src/vision_line/scripts/` then) `D:\Anaconda\envs\opencv39\python.exe -m pytest test_fill_missing_line.py -v`
Expected: 全部 PASS

- [ ] **Step 3: 提交**

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "feat: integrate _fill_missing_line into get_side_line_task_1"
```

---

### Task 5: 使用真实图片集成测试

**Files:**
- Modify: `src/vision_line/scripts/test_fill_missing_line.py`

- [ ] **Step 1: 添加集成测试 — 使用项目中的测试图片验证完整流程**

在 `test_fill_missing_line.py` 末尾追加：

```python
class TestIntegrationWithRealImage:
    """使用真实图片验证完整流程"""

    def _get_test_image(self):
        """加载测试图片并预处理"""
        pictures_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.pardir, "pictures"
        )
        test_img = os.path.join(os.path.join(pictures_dir, "test1.png"))
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
        # 无论是否检测到边线，流程都应正常完成

    def test_midline_via_manual_missing_line(self):
        """手动构造缺失线场景，验证中线计算正确"""
        proc, binary = self._get_test_image()
        img_shape = binary.shape
        # 构造：右线存在，左线缺失
        proc3 = ImageProcess()
        proc3.right_line = [(160, 200), (162, 198), (164, 196)]
        proc3.left_line = []
        result = proc3._fill_missing_line(img_shape)
        assert result is True
        assert len(proc3.supple_left_line) > 0
        assert len(proc3.supple_right_line) > 0
        assert len(proc3.supple_left_line) == len(proc3.supple_right_line)
        # 验证中线可以正确计算
        for lp, rp in zip(proc3.supple_left_line, proc3.supple_right_line):
            assert lp[1] == rp[1]
```

- [ ] **Step 2: 运行所有测试**

Run: (cd `src/vision_line/scripts/` then) `D:\Anaconda\envs\opencv39\python.exe -m pytest test_fill_missing_line.py -v`
Expected: 全部 PASS

- [ ] **Step 3: 提交**

```bash
git add src/vision_line/scripts/test_fill_missing_line.py
git commit -m "test: add integration test for _fill_missing_line with real images"
```
