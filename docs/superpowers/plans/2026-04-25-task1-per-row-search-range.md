# Task1 按行递推搜索窗口 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `get_side_line_task_1()` 的搜索起点从"上一帧边线引导"改为"同帧逐行递推"，并让搜索窗口越往上越窄。

**Architecture:** 在逐行扫描循环内，用 `prev_row_left_x / prev_row_right_x` 递推替代 `_get_search_start_point(prev_*)` 调用；用分段阶梯函数替代固定 `search_range`；增加 miss 计数回退。后处理不变。

**Tech Stack:** Python 3.9, NumPy, OpenCV, pytest

**Spec:** `docs/superpowers/specs/2026-04-25-task1-per-row-search-range-design.md`

**Python interpreter:** `"D:\Anaconda\envs\opencv39\python.exe"`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/vision_line/scripts/image_process.py:613-763` | Modify | `get_side_line_task_1()` — 核心改动 |
| `src/vision_line/scripts/test_per_row_search.py` | Create | 新增单元测试 |
| `src/vision_line/scripts/test_fill_missing_line.py` | No change | 已有测试，运行确认不回归 |

---

### Task 1: 写失败测试 — 验证搜索起点从 mid_x 初始化、递推更新

**Files:**
- Create: `src/vision_line/scripts/test_per_row_search.py`

- [ ] **Step 1: 写测试**

构造一个可控的二值化图像（白底黑边线），调用 `get_side_line_task_1()`，验证：
- `left_line` 和 `right_line` 不为空（说明搜索确实在工作）
- 点的 y 顺序正确（降序）

```python
"""test_per_row_search.py — 按行递推搜索策略单元测试"""
import numpy as np
import pytest
from image_process import ImageProcess


def _make_binary_with_edges(h=240, w=320, left_x=80, right_x=240):
    """生成白底二值图，左、右各一条竖直黑边线（宽3px）"""
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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd src/vision_line/scripts && "D:\Anaconda\envs\opencv39\python.exe" -m pytest test_per_row_search.py -v
```

预期：PASS（因为旧代码也能检测竖直边线）。这个测试确保改造后不回归。

---

### Task 2: 写失败测试 — 验证动态 range 与 miss 回退

**Files:**
- Modify: `src/vision_line/scripts/test_per_row_search.py`

- [ ] **Step 1: 添加 range 和 miss 回退测试**

在 `test_per_row_search.py` 追加：

```python
class TestPerRowSearchRange:
    """验证搜索窗口动态缩小行为"""

    def setup_method(self):
        self.proc = ImageProcess()

    def test_narrow_range_still_finds_edge(self):
        """即使在上半部分（range=30），仍应检测到边线"""
        binary = _make_binary_with_edges(h=240, w=320, left_x=100, right_x=220)
        canvas = np.zeros((240, 320, 3), dtype=np.uint8)
        self.proc.get_side_line_task_1(binary, canvas)
        # 上半部分的点也应该存在（y < 0.6*240=144）
        upper_points = [p for p in self.proc.left_line if p[1] < 144]
        assert len(upper_points) > 0


class TestPerRowSearchFallback:
    """验证 miss 回退行为"""

    def setup_method(self):
        self.proc = ImageProcess()

    def test_one_side_missing_still_produces_supple(self):
        """当一侧完全丢线时，_fill_missing_line 应补出边界线"""
        # 只有右边线的图
        h, w = 240, 320
        img = np.full((h, w), 255, dtype=np.uint8)
        img[:, 239:242] = 0
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        self.proc.get_side_line_task_1(img, canvas)
        # 至少一侧应有 supple 线
        assert len(self.proc.supple_left_line) > 0 or len(self.proc.supple_right_line) > 0
```

- [ ] **Step 2: 运行全部新测试**

```bash
cd src/vision_line/scripts && "D:\Anaconda\envs\opencv39\python.exe" -m pytest test_per_row_search.py -v
```

预期：新测试在旧代码上应 PASS（因为功能目标是兼容的）。

---

### Task 3: 实现 — 重写 get_side_line_task_1() 循环

**Files:**
- Modify: `src/vision_line/scripts/image_process.py:613-763`

这是核心改动。以下是将 `get_side_line_task_1()` 方法体（行 613-763）替换为的新版本。

- [ ] **Step 1: 替换方法体**

将 `get_side_line_task_1` 方法从行 613 开始到 `self._update_prev_frame_lines()`（行 763）之前（即 `finally` 块）的部分替换为：

```python
    def get_side_line_task_1(self, img, canvas, is_draw=False):
        """从图像的中线往两边搜索，获取赛道边线

        使用同帧逐行递推决定搜索起点，搜索窗口越往上越窄。

        Args:
            img: 输入的二值化图像
            canvas: 用于绘制的画布图像
            is_draw: 是否在canvas上绘制调试信息，默认为False
        """
        mid_x = img.shape[1] // 2
        img_h, img_w = img.shape[:2]
        up_ratio = 0.55
        down_ratio = 0.90
        try:
            # 清空列表
            self.left_line.clear()
            self.right_line.clear()
            self.supple_left_line.clear()
            self.supple_right_line.clear()
            self.mid_line.clear()
            self.fit_mid_line.clear()

            # 逐行递推的搜索起点（初始为 mid_x）
            prev_row_left_x = mid_x
            prev_row_right_x = mid_x

            # miss 计数（只在 stable 后计数）
            left_miss_count = 0
            right_miss_count = 0
            miss_threshold = 3

            # 稳定点缓冲区及标志
            left_stable_buf = []
            right_stable_buf = []
            left_stable = [False]
            right_stable = [False]

            diff = np.diff(img == 0, axis=1)  # 计算行内黑白跳变  右-左

            for y in range(
                int(img_h * down_ratio), int(img_h * up_ratio), -1
            ):
                row_diff = diff[y]

                # 动态搜索窗口：二段阶梯
                y_norm = y / img_h
                cur_range = 50 if y_norm > 0.6 else 30

                # --- 左侧赛道线 ---
                search_start_left = max(
                    0, min(prev_row_left_x + self.search_offset, img_w - 1)
                )

                if is_draw:
                    cv2.circle(canvas, (search_start_left, y), 3, (255, 0, 255), -1)

                search_end_left = max(0, search_start_left - cur_range)

                candidates = np.where(row_diff[search_end_left:search_start_left] == 1)[0]
                left_added = False
                if len(candidates) > 0:
                    x = search_end_left + candidates[-1]
                    left_added = self._add_point_with_stable_start(
                        self.left_line,
                        (x, y),
                        left_stable_buf,
                        left_stable,
                        self.x_continual,
                        self.y_continual,
                    )
                    if left_added:
                        prev_row_left_x = x

                # miss 计数（只在 stable 后）
                if left_stable[0] and not left_added:
                    left_miss_count += 1
                else:
                    left_miss_count = 0

                if left_miss_count >= miss_threshold:
                    prev_row_left_x = mid_x
                    left_miss_count = 0

                # --- 右侧赛道线 ---
                search_start_right = max(
                    0, min(prev_row_right_x - self.search_offset, img_w - 1)
                )

                if is_draw:
                    cv2.circle(canvas, (search_start_right, y), 3, (255, 255, 0), -1)

                search_end_right = min(img_w - 1, search_start_right + cur_range)

                candidates = np.where(row_diff[search_start_right:search_end_right] == 1)[0]
                right_added = False
                if len(candidates) > 0:
                    x = search_start_right + candidates[0]
                    right_added = self._add_point_with_stable_start(
                        self.right_line,
                        (x, y),
                        right_stable_buf,
                        right_stable,
                        self.x_continual,
                        self.y_continual,
                    )
                    if right_added:
                        prev_row_right_x = x

                # miss 计数（只在 stable 后）
                if right_stable[0] and not right_added:
                    right_miss_count += 1
                else:
                    right_miss_count = 0

                if right_miss_count >= miss_threshold:
                    prev_row_right_x = mid_x
                    right_miss_count = 0

            # 如果没有丢线，就直接补线；反之要补线
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

            # 使用优化后的边线计算中线
            for y in range(
                min(len(self.supple_left_line), len(self.supple_right_line))
            ):
                mid_x = (
                    self.supple_left_line[y][0] + self.supple_right_line[y][0]
                ) // 2
                mid_y = self.supple_left_line[y][1]
                self.mid_line.append((mid_x, mid_y))

        except Exception as e:
            logging.error(f"Error occurred during getting side lines : {e}")
        finally:
            # 更新上一帧的边线信息
            self._update_prev_frame_lines()
```

- [ ] **Step 2: 运行全部测试确认通过**

```bash
cd src/vision_line/scripts && "D:\Anaconda\envs\opencv39\python.exe" -m pytest test_per_row_search.py test_fill_missing_line.py -v
```

预期：全部 PASS。

---

### Task 4: 运行全部测试 + 确认无回归

**Files:**
- No changes

- [ ] **Step 1: 运行所有已有测试**

```bash
cd src/vision_line/scripts && "D:\Anaconda\envs\opencv39\python.exe" -m pytest test_per_row_search.py test_fill_missing_line.py -v
```

预期：全部 PASS。

- [ ] **Step 2: 提交**

```bash
git add src/vision_line/scripts/image_process.py src/vision_line/scripts/test_per_row_search.py
git commit -m "feat: task1 按行递推搜索窗口，去掉上一帧依赖

- search_start 改为同帧逐行递推（prev_row_left_x/right_x）
- 搜索窗口按 y_norm 分段：>0.6 用50，否则30
- 增加 miss 计数回退（stable 后连续3行未加入则回 mid_x）
- 删除 prev_left/right、is_first_frame 等上一帧相关变量
- 新增 test_per_row_search.py 单元测试"
```

---

### Task 5: 手动可视化验证（需人工操作）

- [ ] 运行 `main_video()` 或 `main()`，进入 `RIGHT_TRACKING/LEFT_TRACKING` 状态
- [ ] 观察左边线（红色点）和右边线（红色点）是否稳定跟踪赛道边缘
- [ ] 观察上方区域是否仍有足够的边线点（不会因缩窗过度丢线）
- [ ] 如有问题，调整 `cur_range` 的分段阈值或 `miss_threshold`
