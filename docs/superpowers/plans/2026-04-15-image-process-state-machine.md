# 图像处理状态机扩展（CROSS 状态）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在已有的 IDLE/TRACKING/CORNER 三状态基础上，扩展 `ProcessState` 枚举和 `main()` 状态机，新增 `CROSS` 状态，并在 `ImageProcess` 类中维护拐点结果和判断/占位函数。

**Architecture:** 在 `ImageProcess` 中新增 `self.left_c`/`self.right_c` 字段存储每帧拐点，新增 `judge_enter_cross_state()` 判断函数和 `run_cross_stage()` 占位函数。在 `main()` 中扩展状态转移逻辑：CORNER 处理完成后检查是否进入 CROSS，CROSS 下 `find_corner=False` 并调用占位函数。

**Tech Stack:** Python 3, OpenCV, enum（标准库）

**Spec:** `docs/superpowers/specs/2026-04-15-image-process-state-machine-design.md`

---

## File Structure

| 操作 | 文件 | 说明 |
|------|------|------|
| Modify | `src/vision_line/scripts/image_process.py` | 唯一改动的文件 |

改动区域：
1. **`ProcessState` 枚举**（第 16-19 行）：新增 `CROSS = 3`
2. **`ImageProcess.__init__`**（第 32-63 行）：新增 `self.left_c`、`self.right_c`
3. **`get_side_line_task_2`**（第 533-791 行）：每帧清空拐点字段，检测到时写入
4. **`ImageProcess` 新增方法**：`judge_enter_cross_state()` 和 `run_cross_stage()`
5. **`main()` 状态机**（第 1085-1199 行）：扩展状态转移 + CROSS 分支

---

### Task 1: 扩展 ProcessState 枚举

**Files:**
- Modify: `src/vision_line/scripts/image_process.py:16-19`

- [ ] **Step 1: 在 ProcessState 中新增 CROSS = 3**

当前代码（第 16-19 行）：
```python
class ProcessState(Enum):
    IDLE = 0  # 未收到指令，不处理
    TRACKING = 1  # 已收到指令，find_corner=False
    CORNER = 2  # 延时已到，find_corner=True
```

改为：
```python
class ProcessState(Enum):
    IDLE = 0  # 未收到指令，不处理
    TRACKING = 1  # 已收到指令，find_corner=False
    CORNER = 2  # 延时已到，find_corner=True
    CROSS = 3  # 双拐点触发，find_corner=False，执行额外处理
```

- [ ] **Step 2: 验证语法**

Run: `python -c "import ast; ast.parse(open('src/vision_line/scripts/image_process.py').read()); print('syntax ok')"`
Expected: `syntax ok`

- [ ] **Step 3: Commit**

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "feat: 扩展 ProcessState 枚举，新增 CROSS 状态"
```

---

### Task 2: 在 ImageProcess 中新增拐点字段 + 判断/占位方法

**Files:**
- Modify: `src/vision_line/scripts/image_process.py:46-63`（`__init__` 末尾）
- Add: `judge_enter_cross_state` 方法（在 `get_side_line_task_1` 之前）
- Add: `run_cross_stage` 方法（紧接 `judge_enter_cross_state` 之后）

- [ ] **Step 1: 在 `__init__` 中新增 `self.left_c` 和 `self.right_c`**

old_string（锚点）：
```python
        self.fit_mid_line = []

        # 上一帧的边线信息（用于指导当前帧搜索）
```

new_string：
```python
        self.fit_mid_line = []

        self.left_c = None  # 本帧左边线拐点 (x, y)，未检测到时为 None
        self.right_c = None  # 本帧右边线拐点 (x, y)，未检测到时为 None

        # 上一帧的边线信息（用于指导当前帧搜索）
```

- [ ] **Step 2: 新增 `judge_enter_cross_state` 和 `run_cross_stage` 方法**

old_string（锚点，`get_side_line_task_1` 的 def 行）：
```python
    def get_side_line_task_1(self, img, canvas, is_draw=False):
```

new_string（在方法前插入两个新方法）：
```python
    def judge_enter_cross_state(self, img_shape, y_ratio=0.9):
        """判断是否满足进入 CROSS 状态的条件

        Args:
            img_shape: 图像形状 (height, width)
            y_ratio: y 坐标阈值比例，默认 0.9

        Returns:
            bool: 满足条件返回 True，否则 False
        """
        if self.left_c is None or self.right_c is None:
            return False
        img_h = img_shape[0]
        return self.left_c[1] >= img_h * y_ratio and self.right_c[1] >= img_h * y_ratio

    def run_cross_stage(self, binary_img, canvas):
        """CROSS 状态的额外处理（占位函数，待实现）

        Args:
            binary_img: 二值化图像
            canvas: 画布图像
        """
        logging.info("CROSS stage triggered — placeholder, not yet implemented")

    def get_side_line_task_1(self, img, canvas, is_draw=False):
```

- [ ] **Step 3: 验证语法**

Run: `python -c "import ast; ast.parse(open('src/vision_line/scripts/image_process.py').read()); print('syntax ok')"`
Expected: `syntax ok`

- [ ] **Step 4: Commit**

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "feat: 新增 left_c/right_c 字段、judge_enter_cross_state 和 run_cross_stage"
```

---

### Task 3: 修改 get_side_line_task_2 以写入拐点结果到 self

**Files:**
- Modify: `src/vision_line/scripts/image_process.py` — `get_side_line_task_2` 方法内两处

- [ ] **Step 1: 在 get_side_line_task_2 的 try 块开头（清空列表处）增加清空拐点字段**

当前代码（第 560-565 行附近，try 块内的 clear 序列）：
```python
            self.left_line.clear()
            self.right_line.clear()
            self.supple_left_line.clear()
            self.supple_right_line.clear()
            self.mid_line.clear()
            self.fit_mid_line.clear()
```

改为：
```python
            self.left_line.clear()
            self.right_line.clear()
            self.supple_left_line.clear()
            self.supple_right_line.clear()
            self.mid_line.clear()
            self.fit_mid_line.clear()
            self.left_c = None
            self.right_c = None
```

- [ ] **Step 2: 在左边线检测到拐点处写入 `self.left_c`**

在 `left_c = (left_prev_x, left_prev_y)` 之后（约第 621-625 行），追加一行写入 `self.left_c`：

当前代码：
```python
                                                left_c = (
                                                    left_prev_x,
                                                    left_prev_y,
                                                )
                                                find_left_corner = True
```

改为：
```python
                                                left_c = (
                                                    left_prev_x,
                                                    left_prev_y,
                                                )
                                                self.left_c = left_c
                                                find_left_corner = True
```

- [ ] **Step 3: 在右边线检测到拐点处写入 `self.right_c`**

在 `right_c = (right_prev_x, right_prev_y)` 之后（约第 690-694 行），追加一行写入 `self.right_c`：

当前代码：
```python
                                                right_c = (
                                                    right_prev_x,
                                                    right_prev_y,
                                                )
                                                find_right_corner = True
```

改为：
```python
                                                right_c = (
                                                    right_prev_x,
                                                    right_prev_y,
                                                )
                                                self.right_c = right_c
                                                find_right_corner = True
```

- [ ] **Step 4: 验证语法**

Run: `python -c "import ast; ast.parse(open('src/vision_line/scripts/image_process.py').read()); print('syntax ok')"`
Expected: `syntax ok`

- [ ] **Step 5: Commit**

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "feat: get_side_line_task_2 每帧清空并写入 left_c/right_c 拐点结果"
```

---

### Task 4: 扩展 main() 状态机 — 新增 CROSS 状态转移和处理逻辑

**Files:**
- Modify: `src/vision_line/scripts/image_process.py:1100-1150`（状态转移 + 图像处理块）

- [ ] **Step 1: 替换状态转移逻辑**

当前代码（第 1100-1116 行的状态转移块）：
```python
                # --- 状态机：状态转移 ---
                if state == ProcessState.IDLE:
                    if command_received:
                        state = ProcessState.TRACKING
                        t0 = time.time()
                        logging.info("State: IDLE -> TRACKING (command received)")
                elif state in (ProcessState.TRACKING, ProcessState.CORNER):
                    if not command_received:
                        state = ProcessState.IDLE
                        t0 = None
                        logging.info("State: -> IDLE (command revoked)")
                    elif state == ProcessState.TRACKING and t0 is not None:
                        if time.time() - t0 >= corner_delay_s:
                            state = ProcessState.CORNER
                            logging.info(
                                f"State: TRACKING -> CORNER (after {corner_delay_s}s)"
                            )
```

替换为：
```python
                # --- 状态机：状态转移 ---
                if state == ProcessState.IDLE:
                    if command_received:
                        state = ProcessState.TRACKING
                        t0 = time.time()
                        logging.info("State: IDLE -> TRACKING (command received)")
                elif state in (ProcessState.TRACKING, ProcessState.CORNER, ProcessState.CROSS):
                    if not command_received:
                        state = ProcessState.IDLE
                        t0 = None
                        logging.info("State: -> IDLE (command revoked)")
                    elif state == ProcessState.TRACKING and t0 is not None:
                        if time.time() - t0 >= corner_delay_s:
                            state = ProcessState.CORNER
                            logging.info(
                                f"State: TRACKING -> CORNER (after {corner_delay_s}s)"
                            )
```

- [ ] **Step 2: 替换图像处理块，增加 CROSS 分支**

> **同帧说明：** CROSS 的 `run_cross_stage` 在首次触发帧就会执行（本帧仍以 `find_corner=True` 做了边线检测，但同时也会调用 `run_cross_stage`）。后续帧则以 `find_corner=False` 运行。这是有意为之的立即响应设计。

当前代码（第 1118-1151 行的 `if state != ProcessState.IDLE:` 块）：
```python
                # --- 非 IDLE 状态才执行图像处理 ---
                if state != ProcessState.IDLE:
                    # 保存当前帧到imgprocess
                    imgprocess.frame = frame

                    # 预处理
                    binary_img = imgprocess.preprocess()
                    if binary_img is None:
                        logging.warning(f"Frame {frame_count} preprocessing failed")
                    else:
                        # 获取用于绘制的画布
                        canvas = imgprocess.return_frame()
                        if canvas is not None:
                            # 根据状态决定 find_corner 参数
                            find_corner = state == ProcessState.CORNER

                            # 获取边线（传入canvas用于绘制调试信息）
                            imgprocess.get_side_line_task_2(
                                binary_img,
                                canvas,
                                is_draw=True,
                                find_corner=find_corner,
                            )

                            # 多项式拟合
                            imgprocess.fit_polynomial()

                            canvas = imgprocess.draw_line(canvas)

                            # 显示二值化图像
                            cv2.imshow("binary", binary_img)

                            # 显示处理结果
                            cv2.imshow("processed_img", canvas)
```

替换为：
```python
                # --- 非 IDLE 状态才执行图像处理 ---
                if state != ProcessState.IDLE:
                    # 保存当前帧到imgprocess
                    imgprocess.frame = frame

                    # 预处理
                    binary_img = imgprocess.preprocess()
                    if binary_img is None:
                        logging.warning(f"Frame {frame_count} preprocessing failed")
                    else:
                        # 获取用于绘制的画布
                        canvas = imgprocess.return_frame()
                        if canvas is not None:
                            # 根据状态决定 find_corner 参数
                            find_corner = state == ProcessState.CORNER

                            # 获取边线（传入canvas用于绘制调试信息）
                            imgprocess.get_side_line_task_2(
                                binary_img,
                                canvas,
                                is_draw=True,
                                find_corner=find_corner,
                            )

                            # CORNER 状态：处理完毕后检查是否进入 CROSS
                            if state == ProcessState.CORNER:
                                if imgprocess.judge_enter_cross_state(binary_img.shape, 0.9):
                                    state = ProcessState.CROSS
                                    logging.info("State: CORNER -> CROSS (dual corner detected)")

                            # CROSS 状态：执行额外处理
                            if state == ProcessState.CROSS:
                                imgprocess.run_cross_stage(binary_img, canvas)

                            # 多项式拟合
                            imgprocess.fit_polynomial()

                            canvas = imgprocess.draw_line(canvas)

                            # 显示二值化图像
                            cv2.imshow("binary", binary_img)

                            # 显示处理结果
                            cv2.imshow("processed_img", canvas)
```

- [ ] **Step 3: 验证语法**

Run: `python -c "import ast; ast.parse(open('src/vision_line/scripts/image_process.py').read()); print('syntax ok')"`
Expected: `syntax ok`

- [ ] **Step 4: Commit**

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "feat: main() 状态机扩展 CROSS 状态转移和处理逻辑"
```

---

### Task 5: 手动验证

本项目无自动化测试框架，使用手动验证。

- [ ] **Step 1: 运行脚本验证状态机日志**

Run: `cd src/vision_line/scripts && python image_process.py`

Expected（观察控制台日志）：
1. 启动后立即看到 `State: IDLE -> TRACKING (command received)`
2. 约 2 秒后看到 `State: TRACKING -> CORNER (after 2.0s)`
3. 当视频中出现双拐点且 y 值满足条件时，看到 `State: CORNER -> CROSS (dual corner detected)`
4. 进入 CROSS 后持续看到 `CROSS stage triggered — placeholder, not yet implemented`
5. 窗口中视频正常显示边线检测效果
6. 按 `q` 可正常退出，按空格可暂停/恢复

- [ ] **Step 2: 测试 command_received=False 场景**

临时将 `command_received = True` 改为 `command_received = False`，再次运行。

Expected：
- 不出现状态转移日志
- 窗口不显示边线处理结果（IDLE 状态跳过处理）
- 按 `q` 仍可正常退出
- 验证后改回 `True`

- [ ] **Step 3: 确认最终代码无报错，恢复默认参数**
