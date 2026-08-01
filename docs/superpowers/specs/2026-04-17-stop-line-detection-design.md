# Stop Line Detection Design

**Date:** 2026-04-17
**Status:** Draft
**Related File:** `src/vision_line/scripts/image_process.py`

## Overview

在 `ProcessState.CROSS` 状态下，检测 ROI 区域内的水平白线（停止线），并根据其 y 坐标位置决定是否进入 `TURNING` 状态。

## Requirements

### Functional Requirements

1. **ROI 定义**（基于归一化坐标）
   - y 范围：`[0.55, 0.80]`
   - x 范围：`[0.30, 0.70]`

2. **检测目标**
   - 检测一条较长的近水平白线
   - 白线可能超出 ROI 边界，但只追踪 ROI 内的部分
   - 返回该白线段的中点坐标

3. **连通性阈值**
   - x 连续性：相邻点 `|Δx| ≤ 4`
   - y 连续性：相邻点 `|Δy| ≤ 2`

4. **种子点规则**
   - x 坐标固定为 ROI 的中点：`seed_x = (roi_x0 + roi_x1) // 2`
   - 从 y 从下往上扫描，在 `[seed_x-4, seed_x+4]` 范围内找第一个白点
   - 找到离 seed_x 最近的白点作为种子点

5. **追踪算法**
   - 使用贪心跟踪：从种子点向左右两侧扩展
   - 左扩展：下一点必须在 `x ∈ [cur_x-4, cur_x-1]` 且 `y ∈ [cur_y-2, cur_y+2]`
   - 右扩展：下一点必须在 `x ∈ [cur_x+1, cur_x+4]` 且 `y ∈ [cur_y-2, cur_y+2]`
   - 在候选窗口内选择最接近水平（|Δy| 最小）的点
   - 到达 ROI 边界或找不到下一个点时停止

6. **有效性判定**
   - 线段 x 跨度 `max_x - min_x` 必须 > 20 像素
   - 否则认为无效，返回 `None`

7. **状态转换**
   - 当停止线中点的归一化 y 坐标 > 0.78 时，进入 `TURNING` 状态

### Non-Functional Requirements

1. **性能**
   - 从下往上扫描，找到第一个停止线就立即返回
   - ROI 较小，单帧处理时间应 < 10ms

2. **鲁棒性**
   - 允许白线轻微弯曲（y±2 容忍）
   - 允许白线超出 ROI（追踪到边界即停）

## Design

### 1. 状态扩展

在 `ProcessState` 枚举中添加新状态：

```python
class ProcessState(Enum):
    IDLE = 0
    TRACKING = 1
    CORNER = 2
    CROSS = 3
    TURNING = 4  # 新增
```

### 2. 新增方法

#### 2.1 `get_stop_line()`

**签名：**
```python
def get_stop_line(self, binary_img, is_draw=False, canvas=None) -> Optional[Tuple[int, int]]
```

**参数：**
- `binary_img`: 二值化图像（numpy array）
- `is_draw`: 是否绘制调试信息
- `canvas`: 绘制画布（BGR 格式）

**返回值：**
- 成功：`(mid_x, mid_y)` - 停止线段中点坐标
- 失败：`None`

**算法步骤：**
0. **输入验证**：若 `binary_img is None`，记录警告日志并返回 `None`

1. 计算 ROI 边界：
   - `roi_y0 = int(h * 0.55)`, `roi_y1 = int(h * 0.80)`
   - `roi_x0 = int(w * 0.30)`, `roi_x1 = int(w * 0.70)`

2. 种子点选择：
   - `seed_x = (roi_x0 + roi_x1) // 2`
   - 从 `y = roi_y1` 到 `roi_y0` 向上扫描
   - 在每个 y 上检查 `x ∈ [seed_x-4, seed_x+4]` 是否有白点
   - 找到则取离 seed_x 最近的白点作为种子点 `(x_seed, y_seed)`
   - 若整段 y 都没有白点，返回 `None`

3. 左侧追踪：
   - 当前点 `cur = (x_seed, y_seed)`
   - 在 `x ∈ [cur_x-4, cur_x-1]`, `y ∈ [cur_y-2, cur_y+2]` 窗口内找白点
   - 找到则选择 |Δy| 最小的点；若 |Δy| 相同，选择 x 最小的点
   - 加入点集，更新 `cur`
   - 找不到或到达 `roi_x0` 时停止

4. 右侧追踪：
   - 重置 `cur = (x_seed, y_seed)`
   - 在 `x ∈ [cur_x+1, cur_x+4]`, `y ∈ [cur_y-2, cur_y+2]` 窗口内找白点
   - 找到则选择 |Δy| 最小的点；若 |Δy| 相同，选择 x 最小的点
   - 加入点集，更新 `cur`
   - 找不到或到达 `roi_x1` 时停止

5. 计算中点：
   - 合并左侧点、种子点、右侧点，按 x 排序
   - `x_span = max_x - min_x`
   - 若 `x_span <= 20`，返回 `None`
   - `mid_x = (min_x + max_x) // 2`
   - `mid_y`：查找点集中 x 坐标最接近 `mid_x` 的点，取其 y 坐标
   - 返回 `(mid_x, mid_y)`

6. 绘制（`is_draw=True` 且 `canvas != None`）：
   - 线段点用黄色小圆点 `(0, 255, 255)` 标记
   - 中点用红色圆点 `(0, 0, 255)` 标记
   - 不画连线，不添加文字

#### 2.2 `should_enter_turning()`

**签名：**
```python
def should_enter_turning(self, stop_mid, img_shape, y_thresh=0.78) -> bool
```

**参数：**
- `stop_mid`: 停止线中点坐标 `(x, y)` 或 `None`
- `img_shape`: 图像形状 `(h, w, ...)`
- `y_thresh`: y 坐标阈值（归一化），默认 0.78

**返回值：**
- `True`: 应该进入 TURNING 状态
- `False`: 不应该进入

**逻辑：**
- 若 `stop_mid is None`，返回 `False`
- 否则计算归一化 y：`y_norm = stop_mid[1] / img_shape[0]`
- 返回 `y_norm > y_thresh`
- 注：当 `y_norm == y_thresh` 时，不进入 TURNING 状态

### 3. 状态机集成

在 `main()` 函数的状态机中，替换现有的空 CROSS 状态处理块（约第 1162-1163 行）：

```python
# CROSS 状态处理
if state == ProcessState.CROSS:
    # 获取停止线
    stop_mid = imgprocess.get_stop_line(binary_img, is_draw=True, canvas=canvas)

    # 检查是否进入 TURNING
    if imgprocess.should_enter_turning(stop_mid, binary_img.shape):
        state = ProcessState.TURNING
        y_norm = stop_mid[1] / binary_img.shape[0]
        logging.info(f"State: CROSS -> TURNING (stop line at y={stop_mid[1]}, y_norm={y_norm:.2f})")
```

### 4. 实现位置

- `ProcessState.TURNING`: 添加到 `ProcessState` 枚举（约第 15-19 行）
- `get_stop_line()`: 添加为 `ImageProcess` 类方法（建议放在 `judge_enter_cross_state()` 之后）
- `should_enter_turning()`: 添加为 `ImageProcess` 类方法（建议放在 `get_stop_line()` 之后）
- 状态转换逻辑: 修改 `main()` 函数（约第 1136-1172 行）

## Error Handling

1. `binary_img is None`: 记录警告日志，返回 `None`
2. ROI 无白点: 返回 `None`
3. 线段过短 (x_span ≤ 20): 返回 `None`
4. `canvas is None` 且 `is_draw=True`: 不绘制，不报错

## Testing

### 单元测试场景
1. ROI 内无白点 → 返回 `None`
2. 白点存在但无法形成有效线段 → 返回 `None`
3. 有效水平白线 → 返回正确中点
4. 白线在 ROI 底部 (y_norm < 0.78) → `should_enter_turning()` 返回 `False`
5. 白线在 ROI 顶部 (y_norm > 0.78) → `should_enter_turning()` 返回 `True`

### 集成测试
- 在 `main()` 中运行视频，验证 CROSS → TURNING 状态转换
- 检查 canvas 绘制是否正确显示线段和中点

### 性能测试
- 使用 `time.perf_counter()` 测量 `get_stop_line()` 执行时间
- 在测试视频中连续处理 100 帧，记录平均处理时间
- 预期：平均处理时间 < 10ms (在 320x240 分辨率下)

## Open Questions

None
