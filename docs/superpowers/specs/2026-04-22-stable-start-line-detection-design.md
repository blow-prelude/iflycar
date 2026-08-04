# 设计：边线检测初始稳定点机制

## 背景

`image_process.py` 中的 `get_side_line_task_1` 和 `get_side_line_task_2` 通过逐行扫描检测左右赛道边线。当前行为：第一个检测到的点无条件加入边线列表；后续点通过连续性检查（距离阈值）后才能加入。

问题：如果第一个点是噪声或误检，后续所有连续性检查都锚定在错误的起点上，导致整条边线质量下降。

## 方案

引入"初始稳定阶段"——前 N 个点（默认 5）必须全部互相连续。如果在此阶段检测到不连续，丢弃缓冲区中的所有点并重新开始积累。只有当 N 个连续点确认后，才将它们提交到正式边线列表。稳定阶段之后，使用现有的连续性检查逻辑不变。

## 变更内容

### 1. 新增类属性

在 `ImageProcess.__init__` 中添加：

```python
self.init_stable_count = 5  # 初始连续点数阈值
```

### 2. 新增辅助方法 `_add_point_with_stable_start`

```python
def _add_point_with_stable_start(self, line, point, stable_buf, x_thresh, y_thresh):
```

**参数：**
- `line`：正式边线列表（`self.left_line` 或 `self.right_line`）
- `point`：当前行找到的候选点 `(x, y)`
- `stable_buf`：临时缓冲区列表（每侧一个，由调用方维护）
- `x_thresh`, `y_thresh`：连续性阈值

**连续性判断**：`abs(last_x - point_x) < x_thresh and abs(last_y - point_y) < y_thresh`（严格小于，与现有代码一致）

**逻辑（伪代码）：**

```python
def _add_point_with_stable_start(self, line, point, stable_buf, x_thresh, y_thresh):
    if len(line) >= self.init_stable_count:
        # 常规期：与 line 末点比较连续性
        if abs(line[-1][0] - point[0]) < x_thresh and abs(line[-1][1] - point[1]) < y_thresh:
            line.append(point)
    else:
        # 稳定期
        if len(stable_buf) == 0:
            stable_buf.append(point)  # 第一个点，无需检查
        elif abs(stable_buf[-1][0] - point[0]) < x_thresh and abs(stable_buf[-1][1] - point[1]) < y_thresh:
            stable_buf.append(point)
            if len(stable_buf) >= self.init_stable_count:
                line.extend(stable_buf)
                stable_buf.clear()
        else:
            # 不连续：丢弃缓冲区，当前点作为新起点
            stable_buf.clear()
            stable_buf.append(point)
```

### 3. 修改 `get_side_line_task_1`

在行扫描循环之前初始化：
```python
left_stable_buf = []
right_stable_buf = []
```

**左边线**（约 559-585 行）：将找到候选点后的整个 `if len(self.left_line) == 0 ... else ...` 判断块替换为辅助方法调用。**保留** `break` 语句——找到边线跳变后仍需 `break` 退出像素搜索循环，只是将点添加逻辑替换为辅助方法。同时移除 `found_left` 标志及其相关日志。

```python
# 原来的 if/else 块替换为：
self._add_point_with_stable_start(self.left_line, (i, j), left_stable_buf, 50, 50)
break  # 找到边线后仍需退出像素搜索循环
```

**右边线**（约 604-634 行）：同理替换，保留 `break`。

```python
self._add_point_with_stable_start(self.right_line, (i, j), right_stable_buf, 50, 50)
break
```

移除不再需要的 `found_left` / `found_right` 标志及相关日志。

### 4. 修改 `get_side_line_task_2`

在行扫描循环之前初始化：
```python
left_stable_buf = []
right_stable_buf = []
```

将左边线点添加逻辑（约 733-747 行）替换为：
```python
self._add_point_with_stable_start(
    self.left_line, (x, y), left_stable_buf, self.x_continual, self.y_continual
)
```

将右边线点添加逻辑（约 776-789 行）替换为：
```python
self._add_point_with_stable_start(
    self.right_line, (x, y), right_stable_buf, self.x_continual, self.y_continual
)
```

## 不受影响的部分

- 拐点检测（task_2 中的 `find_corner` 逻辑）：在点添加之前独立执行，不受影响
- 注意：拐点检测的 `left_pre_p`/`left_cur_p`/`left_nxt_p` 跟踪变量会记录稳定阶段被丢弃的点，但这些仅用于角度计算，不影响 `left_line`/`right_line` 的内容
- 线性插值、边界填充、中线计算：操作最终 `left_line`/`right_line` 结果，不受影响
- `_update_prev_frame_lines`：不变
- `draw_line`、`fit_polynomial`：不变

## 边界情况

- **帧内候选点不足**：如果整帧扫描结束后稳定缓冲区未满（少于 N 个候选点），`left_line`/`right_line` 保持为空。下游的 `_linear_interpolation`、`_fill_boundary`、中线计算均已对空列表做保护，不会崩溃。
- **稳定缓冲区在函数调用间不复用**：`stable_buf` 是每次调用 `get_side_line_task_*` 时新初始化的局部变量，帧间不会残留旧数据。

## 阈值对照

| 函数 | 左侧 x/y 阈值 | 右侧 x/y 阈值 |
|------|--------------|--------------|
| `task_1` | 50, 50 | 50, 50 |
| `task_2` | `self.x_continual`, `self.y_continual` | `self.x_continual`, `self.y_continual` |

与变更前各函数使用的阈值一致。
