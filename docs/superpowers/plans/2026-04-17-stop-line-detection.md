# Stop Line Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 ProcessState.CROSS 状态下检测 ROI 区域内的水平白线，并根据其 y 坐标决定是否进入 TURNING 状态

**Architecture:** 在 ImageProcess 类中添加 get_stop_line() 方法（贪心跟踪算法）和 should_enter_turning() 判断方法，在 ProcessState 枚举中添加 TURNING 状态，修改 main() 函数的状态机逻辑

**Tech Stack:** Python, OpenCV (cv2), NumPy

---

## File Structure

**Modified Files:**
- `src/vision_line/scripts/image_process.py` - 主要实现文件
  - ProcessState 枚举（约第 15-19 行）
  - ImageProcess 类方法（在 judge_enter_cross_state() 之后添加）
  - main() 函数状态机逻辑（约第 1136-1172 行）

---

## Task 1: 添加 TURNING 状态到枚举

**Files:**
- Modify: `src/vision_line/scripts/image_process.py:15-19`

- [ ] **Step 1: 添加 TURNING 状态**

在 ProcessState 枚举中添加新状态：

```python
class ProcessState(Enum):
    IDLE = 0
    TRACKING = 1
    CORNER = 2
    CROSS = 3
    TURNING = 4  # 新增：检测到停止线后的转弯状态
```

- [ ] **Step 2: 验证代码语法**

运行: `python -m py_compile src/vision_line/scripts/image_process.py`
预期: 无语法错误

- [ ] **Step 3: 提交**

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "feat: add TURNING state to ProcessState enum"
```

---

## Task 2: 实现 should_enter_turning() 方法

**Files:**
- Modify: `src/vision_line/scripts/image_process.py` (在 judge_enter_cross_state() 方法之后添加)

- [ ] **Step 1: 写测试代码（先写测试）**

创建临时测试文件验证逻辑：

```python
# 测试用例
def test_should_enter_turning():
    # 创建 ImageProcess 实例
    imgprocess = ImageProcess()

    # 测试 1: stop_mid 为 None
    assert imgprocess.should_enter_turning(None, (240, 320)) == False

    # 测试 2: y_norm < 0.78
    assert imgprocess.should_enter_turning((100, 180), (240, 320)) == False  # 180/240=0.75

    # 测试 3: y_norm > 0.78
    assert imgprocess.should_enter_turning((100, 200), (240, 320)) == True  # 200/240=0.833

    # 测试 4: y_norm == 0.78 (边界情况)
    assert imgprocess.should_enter_turning((100, 187), (240, 320)) == False  # 187/240≈0.779

    print("所有测试通过!")
```

- [ ] **Step 2: 运行测试验证失败**

运行: `python -c "from src.vision_line.scripts.image_process import ImageProcess; test_should_enter_turning()"`
预期: AttributeError: should_enter_turning 方法不存在

- [ ] **Step 3: 实现 should_enter_turning() 方法**

在 `judge_enter_cross_state()` 方法之后添加：

```python
def should_enter_turning(self, stop_mid, img_shape, y_thresh=0.78):
    """判断是否应该进入 TURNING 状态

    Args:
        stop_mid: 停止线中点坐标 (x, y) 或 None
        img_shape: 图像形状 (h, w, ...)
        y_thresh: y 坐标阈值（归一化），默认 0.78

    Returns:
        bool: True 表示应该进入 TURNING 状态
    """
    if stop_mid is None:
        return False

    y_norm = stop_mid[1] / img_shape[0]
    return y_norm > y_thresh
```

- [ ] **Step 4: 运行测试验证通过**

运行: `python -c "from src.vision_line.scripts.image_process import ImageProcess; test_should_enter_turning()"`
预期: 输出 "所有测试通过!"

- [ ] **Step 5: 清理测试代码并提交**

删除或注释掉临时测试代码，然后提交：

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "feat: add should_enter_turning() method"
```

---

## Task 3: 实现 get_stop_line() 方法 - 基础框架

**Files:**
- Modify: `src/vision_line/scripts/image_process.py` (在 should_enter_turning() 方法之后添加)

- [ ] **Step 1: 写基础框架测试**

```python
def test_get_stop_line_basic():
    import numpy as np
    imgprocess = ImageProcess()

    # 创建空白测试图像 (240x320)
    binary_img = np.zeros((240, 320), dtype=np.uint8)

    # 测试 1: 空白图像应该返回 None
    result = imgprocess.get_stop_line(binary_img)
    assert result is None, f"Expected None, got {result}"

    # 测试 2: None 输入应该返回 None
    result = imgprocess.get_stop_line(None)
    assert result is None, f"Expected None, got {result}"

    print("基础框架测试通过!")
```

- [ ] **Step 2: 运行测试验证失败**

运行: `python -c "from src.vision_line.scripts.image_process import ImageProcess; import numpy as np; test_get_stop_line_basic()"`
预期: AttributeError: get_stop_line 方法不存在

- [ ] **Step 3: 实现基础框架（方法签名和输入验证）**

```python
def get_stop_line(self, binary_img, is_draw=False, canvas=None):
    """在 ROI 内检测水平白线并返回其中点

    Args:
        binary_img: 二值化图像 (numpy array)
        is_draw: 是否绘制调试信息
        canvas: 绘制画布 (BGR 格式)

    Returns:
        Optional[Tuple[int, int]]: 成功返回 (mid_x, mid_y)，失败返回 None
    """
    # 输入验证
    if binary_img is None:
        logging.warning("binary_img is None")
        return None

    h, w = binary_img.shape[:2]

    # 计算 ROI 边界
    roi_y0 = int(h * 0.55)
    roi_y1 = int(h * 0.80)
    roi_x0 = int(w * 0.30)
    roi_x1 = int(w * 0.70)

    logging.debug(f"ROI: y=[{roi_y0}, {roi_y1}], x=[{roi_x0}, {roi_x1}]")

    # TODO: 后续实现检测逻辑
    return None
```

- [ ] **Step 4: 运行测试验证通过**

运行: `python -c "from src.vision_line.scripts.image_process import ImageProcess; import numpy as np; test_get_stop_line_basic()"`
预期: 输出 "基础框架测试通过!"

- [ ] **Step 5: 提交**

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "feat: add get_stop_line() basic framework with input validation"
```

---

## Task 4: 实现种子点查找逻辑

**Files:**
- Modify: `src/vision_line/scripts/image_process.py:get_stop_line()`

- [ ] **Step 1: 写种子点查找测试**

```python
def test_seed_point_finding():
    import numpy as np
    imgprocess = ImageProcess()

    # 创建测试图像：在 ROI 中点位置画一条水平白线
    binary_img = np.zeros((240, 320), dtype=np.uint8)
    seed_x = (int(320*0.30) + int(320*0.70)) // 2  # ROI 中点 x
    seed_y = int(240 * 0.70)  # 在 y=0.70 位置

    # 画水平白线（x 范围足够长）
    for x in range(int(320*0.30), int(320*0.70)):
        binary_img[seed_y, x] = 255

    result = imgprocess.get_stop_line(binary_img, is_draw=False)
    assert result is not None, "应该能找到停止线"
    print(f"种子点测试通过! 找到中点: {result}")
```

- [ ] **Step 2: 运行测试验证失败**

运行: `python -c "from src.vision_line.scripts.image_process import ImageProcess; import numpy as np; test_seed_point_finding()"`
预期: 返回 None（尚未实现检测逻辑）

- [ ] **Step 3: 实现种子点查找**

在 `get_stop_line()` 方法中，替换 `# TODO` 部分：

```python
    # 计算 ROI 边界
    roi_y0 = int(h * 0.55)
    roi_y1 = int(h * 0.80)
    roi_x0 = int(w * 0.30)
    roi_x1 = int(w * 0.70)

    logging.debug(f"ROI: y=[{roi_y0}, {roi_y1}], x=[{roi_x0}, {roi_x1}]")

    # 种子点选择：x 固定为 ROI 中点
    seed_x = (roi_x0 + roi_x1) // 2

    # 从下往上扫描，在 [seed_x-4, seed_x+4] 范围内找第一个白点
    x_seed, y_seed = None, None
    for y in range(roi_y1, roi_y0 - 1, -1):
        # 在 seed_x 附近搜索
        search_start = max(roi_x0, seed_x - 4)
        search_end = min(roi_x1, seed_x + 4)

        for x in range(search_start, search_end + 1):
            if binary_img[y, x] == 255:
                # 找到白点，记录最接近 seed_x 的点
                if x_seed is None or abs(x - seed_x) < abs(x_seed - seed_x):
                    x_seed, y_seed = x, y

        # 如果这一行找到了白点，选择最接近 seed_x 的作为种子点
        if x_seed is not None:
            break

    if x_seed is None:
        logging.debug("未在 ROI 内找到种子点")
        return None

    logging.debug(f"种子点: ({x_seed}, {y_seed})")
```

- [ ] **Step 4: 运行测试验证通过**

运行: `python -c "from src.vision_line.scripts.image_process import ImageProcess; import numpy as np; test_seed_point_finding()"`
预期: 输出 "种子点测试通过!" 并显示找到的中点

- [ ] **Step 5: 提交**

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "feat: implement seed point finding in get_stop_line()"
```

---

## Task 5: 实现左右追踪扩展逻辑

**Files:**
- Modify: `src/vision_line/scripts/image_process.py:get_stop_line()`

- [ ] **Step 1: 写追踪逻辑测试**

```python
def test_line_tracking():
    import numpy as np
    imgprocess = ImageProcess()

    # 创建测试图像：画一条有轻微 y 变化的长白线
    binary_img = np.zeros((240, 320), dtype=np.uint8)
    roi_x0, roi_x1 = int(320*0.30), int(320*0.70)
    base_y = int(240 * 0.70)

    # 画一条带有 y 抖动的白线（测试连通性容忍）
    for x in range(roi_x0, roi_x1):
        y_offset = (x % 5) - 2  # -2 到 +2 的抖动
        y = base_y + y_offset
        if 0 <= y < 240:
            binary_img[y, x] = 255

    result = imgprocess.get_stop_line(binary_img, is_draw=False)
    assert result is not None, "应该能找到停止线"
    mid_x, mid_y = result
    expected_mid_x = (roi_x0 + roi_x1) // 2
    assert abs(mid_x - expected_mid_x) < 5, f"中点 x 坐标偏差过大: {mid_x} vs {expected_mid_x}"
    print(f"追踪测试通过! 中点: ({mid_x}, {mid_y})")
```

- [ ] **Step 2: 运行测试验证失败**

运行: `python -c "from src.vision_line.scripts.image_process import ImageProcess; import numpy as np; test_line_tracking()"`
预期: 找不到线段或线段过短（尚未实现追踪逻辑）

- [ ] **Step 3: 实现左侧追踪**

在种子点查找之后添加：

```python
    logging.debug(f"种子点: ({x_seed}, {y_seed})")

    # 左侧追踪
    left_points = []
    cur_x, cur_y = x_seed, y_seed

    while True:
        found_next = False
        best_x, best_y = None, None
        best_dy = float('inf')

        # 在候选窗口内找下一个点
        for next_x in range(max(roi_x0, cur_x - 4), cur_x):
            for next_y in range(max(0, cur_y - 2), min(h, cur_y + 3)):
                if binary_img[next_y, next_x] == 255:
                    dy = abs(next_y - cur_y)
                    # 选择 |Δy| 最小的点；若相同，选择 x 最小的
                    if dy < best_dy or (dy == best_dy and (best_x is None or next_x < best_x)):
                        best_dy = dy
                        best_x, best_y = next_x, next_y
                        found_next = True

        if found_next:
            left_points.append((best_x, best_y))
            cur_x, cur_y = best_x, best_y
        else:
            break
```

- [ ] **Step 4: 实现右侧追踪**

在左侧追踪之后添加：

```python
    # 右侧追踪
    right_points = []
    cur_x, cur_y = x_seed, y_seed

    while True:
        found_next = False
        best_x, best_y = None, None
        best_dy = float('inf')

        # 在候选窗口内找下一个点
        for next_x in range(cur_x + 1, min(roi_x1, cur_x + 5)):
            for next_y in range(max(0, cur_y - 2), min(h, cur_y + 3)):
                if binary_img[next_y, next_x] == 255:
                    dy = abs(next_y - cur_y)
                    # 选择 |Δy| 最小的点；若相同，选择 x 最小的
                    if dy < best_dy or (dy == best_dy and (best_x is None or next_x < best_x)):
                        best_dy = dy
                        best_x, best_y = next_x, next_y
                        found_next = True

        if found_next:
            right_points.append((best_x, best_y))
            cur_x, cur_y = best_x, best_y
        else:
            break
```

- [ ] **Step 5: 实现中点计算和有效性判定**

在右侧追踪之后添加：

```python
    # 合并所有点并计算中点
    all_points = list(reversed(left_points)) + [(x_seed, y_seed)] + right_points

    if len(all_points) == 0:
        return None

    # 计算线段跨度
    min_x = min(p[0] for p in all_points)
    max_x = max(p[0] for p in all_points)
    x_span = max_x - min_x

    # 有效性判定：x 跨度必须 > 20
    if x_span <= 20:
        logging.debug(f"线段过短: x_span={x_span} <= 20")
        return None

    # 计算中点
    mid_x = (min_x + max_x) // 2

    # 找到 x 最接近 mid_x 的点的 y 坐标
    mid_y = min(all_points, key=lambda p: abs(p[0] - mid_x))[1]

    logging.debug(f"检测到停止线: 中点=({mid_x}, {mid_y}), x_span={x_span}, 点数={len(all_points)}")

    # 绘制（如果需要）
    if is_draw and canvas is not None:
        # 画线段点（黄色）
        for point in all_points:
            cv2.circle(canvas, point, 2, (0, 255, 255), -1)
        # 画中点（红色）
        cv2.circle(canvas, (mid_x, mid_y), 4, (0, 0, 255), -1)

    return (mid_x, mid_y)
```

- [ ] **Step 6: 运行测试验证通过**

运行: `python -c "from src.vision_line.scripts.image_process import ImageProcess; import numpy as np; test_line_tracking()"`
预期: 输出 "追踪测试通过!"

- [ ] **Step 7: 提交**

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "feat: implement line tracking in get_stop_line()"
```

---

## Task 6: 集成到状态机

**Files:**
- Modify: `src/vision_line/scripts/image_process.py:main()` 函数

- [ ] **Step 1: 定位现有 CROSS 状态处理代码**

查找 main() 函数中的 CROSS 状态处理块（约第 1162-1163 行）。

- [ ] **Step 2: 替换 CROSS 状态处理逻辑**

将现有的空 CROSS 处理块替换为：

```python
                # CROSS 状态：检测停止线
                if state == ProcessState.CROSS:
                    # 获取停止线
                    stop_mid = imgprocess.get_stop_line(binary_img, is_draw=True, canvas=canvas)

                    # 检查是否进入 TURNING
                    if imgprocess.should_enter_turning(stop_mid, binary_img.shape):
                        state = ProcessState.TURNING
                        y_norm = stop_mid[1] / binary_img.shape[0]
                        logging.info(f"State: CROSS -> TURNING (stop line at y={stop_mid[1]}, y_norm={y_norm:.2f})")
```

- [ ] **Step 3: 验证代码语法**

运行: `python -m py_compile src/vision_line/scripts/image_process.py`
预期: 无语法错误

- [ ] **Step 4: 提交**

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "feat: integrate stop line detection into state machine (CROSS -> TURNING)"
```

---

## Task 7: 集成测试与验证

**Files:**
- Test: 运行 main() 函数使用测试视频

- [ ] **Step 1: 运行主程序测试**

运行: `python src/vision_line/scripts/image_process.py`
预期: 程序正常启动，显示视频处理窗口

- [ ] **Step 2: 验证状态转换**

在视频播放中观察：
1. 状态应该从 IDLE -> TRACKING -> CORNER -> CROSS
2. 当检测到停止线且 y_norm > 0.78 时，状态应该进入 TURNING
3. canvas 上应该显示检测到的停止线（黄色点）和中点（红色点）

- [ ] **Step 3: 验证日志输出**

检查控制台日志，应该包含：
- "检测到停止线: 中点=(...), x_span=..."
- "State: CROSS -> TURNING (stop line at y=..., y_norm=...)"

- [ ] **Step 4: 性能验证**

观察 FPS 日志，确保处理速度没有明显下降（预期平均处理时间 < 10ms）

- [ ] **Step 5: 清理并最终提交**

如果有调试代码，清理后提交：

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "test: verify stop line detection integration"
```

---

## Summary

完成以上 7 个任务后，将实现：
1. ✅ ProcessState.TURNING 新状态
2. ✅ get_stop_line() 方法：ROI 内检测水平白线
3. ✅ should_enter_turning() 方法：判断是否进入 TURNING
4. ✅ 状态机集成：CROSS -> TURNING 转换

关键参数：
- ROI: y∈[0.55, 0.80], x∈[0.30, 0.70]
- 连通性: |Δx|≤4, |Δy|≤2
- 最小线段长度: x_span > 20
- 状态转换阈值: y_norm > 0.78
