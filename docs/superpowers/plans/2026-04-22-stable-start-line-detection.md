# 边线检测初始稳定点机制 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `get_side_line_task_1` 和 `get_side_line_task_2` 添加初始稳定点检测，前 N 个点必须全部连续才能正式加入边线列表。

**Architecture:** 提取公共辅助方法 `_add_point_with_stable_start`，封装"稳定期缓冲 → 常规期直加"逻辑。两个 task 函数各自在行扫描循环前初始化局部缓冲区，在找到候选点后调用该方法。

**Tech Stack:** Python 3, OpenCV, NumPy

**Spec:** `docs/superpowers/specs/2026-04-22-stable-start-line-detection-design.md`

---

## File Structure

| Action | File | 职责 |
|--------|------|------|
| Modify | `src/vision_line/scripts/image_process.py` | 唯一需要修改的文件 |

---

### Task 1: 新增类属性和辅助方法

**Files:**
- Modify: `src/vision_line/scripts/image_process.py`

- [ ] **Step 1: 在 `__init__` 中添加 `init_stable_count` 属性**

在 `src/vision_line/scripts/image_process.py:71`（`self.search_offset = 30` 之后）添加：

```python
        self.init_stable_count = 5  # 初始连续点数阈值
```

- [ ] **Step 2: 添加 `_add_point_with_stable_start` 方法**

在 `_update_prev_frame_lines` 方法（第 278 行）之前添加新方法：

```python
    def _add_point_with_stable_start(self, line, point, stable_buf, x_thresh, y_thresh):
        """初始稳定点检测：前 N 个点必须全部连续才加入列表

        Args:
            line: 正式边线列表
            point: 候选点 (x, y)
            stable_buf: 稳定期临时缓冲区（由调用方维护）
            x_thresh: x 方向连续性阈值
            y_thresh: y 方向连续性阈值
        """
        if len(line) >= self.init_stable_count:
            # 常规期：与 line 末点比较连续性
            if abs(line[-1][0] - point[0]) < x_thresh and abs(line[-1][1] - point[1]) < y_thresh:
                line.append(point)
        else:
            # 稳定期
            if len(stable_buf) == 0:
                stable_buf.append(point)
            elif abs(stable_buf[-1][0] - point[0]) < x_thresh and abs(stable_buf[-1][1] - point[1]) < y_thresh:
                stable_buf.append(point)
                if len(stable_buf) >= self.init_stable_count:
                    line.extend(stable_buf)
                    stable_buf.clear()
            else:
                stable_buf.clear()
                stable_buf.append(point)
```

- [ ] **Step 3: Commit**

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "feat: 新增 init_stable_count 属性和 _add_point_with_stable_start 辅助方法"
```

---

### Task 2: 修改 `get_side_line_task_1`

**Files:**
- Modify: `src/vision_line/scripts/image_process.py:500-661`

- [ ] **Step 1: 在行扫描循环前添加缓冲区初始化**

在 `src/vision_line/scripts/image_process.py:534`（`is_first_frame = ...` 之后、`for j in range(...)` 之前）添加：

```python
            # 稳定点缓冲区
            left_stable_buf = []
            right_stable_buf = []
```

- [ ] **Step 2: 替换左边线点添加逻辑**

将 `src/vision_line/scripts/image_process.py:559-585`（从 `found_left = False` 到 `if not found_left and not is_first_frame:` 的日志行）全部替换为：

```python
                for i in range(search_start, search_end_left, -1):
                    if i <= 1:
                        break
                    if img[j, i] == 0 and img[j, i - 1] != 0:
                        logging.debug(
                            f"find left line at {i}, {j} , value : {img[j, i]}"
                        )
                        self._add_point_with_stable_start(
                            self.left_line, (i, j), left_stable_buf, 50, 50
                        )
                        break
```

注意：
- 移除了 `found_left` 变量及其所有引用
- `break` 保留在找到跳变后执行
- `self._add_point_with_stable_start` 调用替代了原来的 `if len(self.left_line) == 0 ... else ...` 块

- [ ] **Step 3: 替换右边线点添加逻辑**

将 `src/vision_line/scripts/image_process.py` 中原来的 `found_right = False` 到 `if not found_right and not is_first_frame:` 日志行（紧接上一步修改后的代码）全部替换为：

```python
                for i in range(search_start, search_end_right, 1):
                    if i >= img.shape[1] - 1:
                        break
                    if img[j, i] == 0 and img[j, i + 1] != 0:
                        logging.debug(
                            f"find right line at {i}, {j} , value : {img[j, i]}"
                        )
                        self._add_point_with_stable_start(
                            self.right_line, (i, j), right_stable_buf, 50, 50
                        )
                        break
```

- [ ] **Step 4: Commit**

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "feat: get_side_line_task_1 使用稳定点检测替代首个点无条件加入"
```

---

### Task 3: 修改 `get_side_line_task_2`

**Files:**
- Modify: `src/vision_line/scripts/image_process.py:663-821`

- [ ] **Step 1: 在行扫描循环前添加缓冲区初始化**

在 `src/vision_line/scripts/image_process.py:696`（`diff = np.diff(...)` 之前）添加：

```python
            # 稳定点缓冲区
            left_stable_buf = []
            right_stable_buf = []
```

- [ ] **Step 2: 替换左边线点添加逻辑**

将 `src/vision_line/scripts/image_process.py:733-747`（从 `# 如果不是拐点，正常添加到边线中` 注释开始的整个 `if len(self.left_line) == 0:` 块）替换为：

```python
                    # 添加到边线（稳定点检测）
                    self._add_point_with_stable_start(
                        self.left_line, (x, y), left_stable_buf,
                        self.x_continual, self.y_continual
                    )
```

- [ ] **Step 3: 替换右边线点添加逻辑**

将 `src/vision_line/scripts/image_process.py:776-789`（从 `if len(self.right_line) == 0:` 开始的整个块）替换为：

```python
                    # 添加到边线（稳定点检测）
                    self._add_point_with_stable_start(
                        self.right_line, (x, y), right_stable_buf,
                        self.x_continual, self.y_continual
                    )
```

- [ ] **Step 4: Commit**

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "feat: get_side_line_task_2 使用稳定点检测替代首个点无条件加入"
```

---

### Task 4: 验证

- [ ] **Step 1: 运行 main_video() 验证**

在终端中运行：
```bash
cd d:/programs/ucar_ws && python src/vision_line/scripts/image_process.py
```

选择 `main_video()` 入口，确认：
1. 左右边线仍然正常检测（红/绿点绘制正常）
2. 中线（蓝点）仍然正常计算
3. 无报错或异常日志

- [ ] **Step 2: 调整参数测试**

临时将 `self.init_stable_count` 改为 3 或 8，观察检测效果变化，确认参数可调。测试后改回 5。
