# 缺失边线边界填充设计

## 概述

当 `get_side_line_task_1()` 检测不到某一边的赛道线（左边线或右边线为空）时，用图像边界替代缺失边线，使中线计算仍能正常进行。

## 背景

`get_side_line_task_1()` 从图像中线向两侧搜索左右赛道边线。在某些帧中，某一侧可能完全检测不到（例如车辆非常靠近一侧）。当前中线计算仅在两条线都存在时才能工作。

## 设计

### 新方法：`_fill_missing_line(self, img_shape) -> bool`

位于 `ImageProcess` 类中，放在 `_fill_boundary` 方法附近。

**行为：**

- 若 `self.left_line` 为空且 `self.right_line` 非空：
  1. 对 `self.right_line` 执行 `_linear_interpolation()`，得到插值后的右边线
  2. 对插值后的右边线执行底部边界填充（与 `_fill_boundary` 的底部延伸逻辑一致），使其从 `img_h - 1` 开始
  3. 取填充后右边线的 y 范围 `[y_top, bottom_y]`
  4. 从 `bottom_y` 到 `y_top`，步长 -2，生成边界点 `(0, y)`，赋给 `self.supple_left_line`
  5. 将填充后的右边线赋给 `self.supple_right_line`
  6. 返回 `True`

- 若 `self.right_line` 为空且 `self.left_line` 非空：
  1. 对 `self.left_line` 执行 `_linear_interpolation()`，得到插值后的左边线
  2. 对插值后的左边线执行底部边界填充，使其从 `img_h - 1` 开始
  3. 取填充后左边线的 y 范围 `[y_top, bottom_y]`
  4. 从 `bottom_y` 到 `y_top`，步长 -2，生成边界点 `(img_w - 1, y)`，赋给 `self.supple_right_line`
  5. 将填充后的左边线赋给 `self.supple_left_line`
  6. 返回 `True`

- 若两条线都为空或都存在：返回 `False`

**关键设计要点：**
- 已存在的那条线必须先经过 `_linear_interpolation()` 和底部边界填充，以确保两边线 y 坐标对齐、间距一致
- 生成的边界线点使用与填充后存在线相同的 y 范围，保证 `supple_left_line` 和 `supple_right_line` 长度一致
- 边界线点按 y 降序排列（从底部到顶部），与现有代码的点序约定一致

**参数：**
- `img_shape`: 元组 `(height, width, ...)` — 图像形状，用于确定边界坐标

**返回值：** `bool` — 是否填充了缺失线

### 集成到 `get_side_line_task_1()`

**替换**现有的 `if len(self.left_line) > 0 and len(self.right_line) > 0:` 代码块（第 664-672 行）为：

```python
filled = self._fill_missing_line(img.shape)
if not filled:
    # 正常流程：两条线都检测到，执行插值 + 填充边界
    if len(self.left_line) > 0 and len(self.right_line) > 0:
        self.supple_left_line = self._linear_interpolation(self.left_line)
        self.supple_right_line = self._linear_interpolation(self.right_line)
        self.supple_left_line, self.supple_right_line = self._fill_boundary(
            self.supple_left_line, self.supple_right_line, img.shape
        )
```

当 `filled` 为 `True` 时，`_fill_missing_line` 已经填充了 `supple_left_line` 和 `supple_right_line`，后续中线计算正常执行。

### 边界情况

- **两条线都为空**：`_fill_missing_line` 返回 `False`，插值代码块被跳过（与当前行为一致），中线为空。
- **两条线都存在**：`_fill_missing_line` 返回 `False`，正常插值流程不变。

### 对后续帧的影响

`_update_prev_frame_lines()` 在 `finally` 块中会将 `supple_left_line` / `supple_right_line` 保存到 `prev_supple_*` 供下一帧搜索使用。当本帧使用了边界替代点时，这些合成坐标会成为下一帧的搜索起点。这可能导致下一帧搜索偏向图像边缘，但这是可接受的行为——如果车辆仍在该侧，搜索应该从边缘附近开始。
