# 状态机多信号修改 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `image_process.py` 中状态机的单一信号 `command_received` 替换为三个互斥信号，新增 RIGHT_TRACKING 和 LEFT_TRACKING 状态。

**Architecture:** 在 `ProcessState` 枚举中重命名 TRACKING 为 STRAIGHT_TRACKING 并添加两个新状态。`main()` 和 `main_video()` 中的 IDLE 分支改为三路 if-elif，非 IDLE 分支仅排除 RIGHT_TRACKING 和 LEFT_TRACKING，其余状态（STRAIGHT_TRACKING / CORNER / CROSS / TURNING）保持原有图像处理逻辑不变。

**Tech Stack:** Python, OpenCV, NumPy

**Spec:** `docs/superpowers/specs/2026-04-22-state-machine-signals-design.md`

---

### Task 1: 修改 ProcessState 枚举

**Files:**
- Modify: `src/vision_line/scripts/image_process.py:18-23`

- [ ] **Step 1: 重命名 TRACKING 并添加新状态**

将：
```python
class ProcessState(Enum):
    IDLE = 0  # 未收到指令，不处理
    TRACKING = 1  # 已收到指令，find_corner=False
    CORNER = 2  # 延时已到，find_corner=True
    CROSS = 3  # 双拐点触发，find_corner=False，执行额外处理
    TURNING = 4  # 新增：检测到停止线后的转弯状态
```

改为：
```python
class ProcessState(Enum):
    IDLE = 0  # 未收到指令，不处理
    STRAIGHT_TRACKING = 1  # 直行循迹，find_corner=False
    RIGHT_TRACKING = 5  # 右转循迹（占位）
    LEFT_TRACKING = 6  # 左转循迹（占位）
    CORNER = 2  # 延时已到，find_corner=True
    CROSS = 3  # 双拐点触发，find_corner=False，执行额外处理
    TURNING = 4  # 检测到停止线后的转弯状态
```

- [ ] **Step 2: 更新文件中所有对 ProcessState.TRACKING 的引用**

使用 replace_all 将 `ProcessState.TRACKING` 替换为 `ProcessState.STRAIGHT_TRACKING`。

注意：日志中的字符串 `"TRACKING"` 不在此步处理范围内，将在 Task 2/3 中随代码块重写一并更新。

- [ ] **Step 3: 提交**

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "refactor: 重命名 TRACKING 为 STRAIGHT_TRACKING，新增 RIGHT_TRACKING/LEFT_TRACKING 枚举"
```

---

### Task 2: 修改 main_video() 状态机

**Files:**
- Modify: `src/vision_line/scripts/image_process.py:1124` (信号变量)
- Modify: `src/vision_line/scripts/image_process.py:1151-1224` (状态转移)

- [ ] **Step 1: 替换信号变量**

将第 1124 行：
```python
command_received = True
```
改为：
```python
straight_received = True
right_received = False
left_received = False
```

- [ ] **Step 2: 修改 IDLE 分支为三路 if-elif**

将 IDLE 分支：
```python
if state == ProcessState.IDLE:
    if command_received:
        state = ProcessState.STRAIGHT_TRACKING
        t0 = time.time()
        logging.info("State: IDLE -> TRACKING (command received)")
```
改为：
```python
if state == ProcessState.IDLE:
    if straight_received:
        state = ProcessState.STRAIGHT_TRACKING
        t0 = time.time()
        logging.info("State: IDLE -> STRAIGHT_TRACKING (straight received)")
    elif right_received:
        state = ProcessState.RIGHT_TRACKING
        logging.info("State: IDLE -> RIGHT_TRACKING (right received)")
    elif left_received:
        state = ProcessState.LEFT_TRACKING
        logging.info("State: IDLE -> LEFT_TRACKING (left received)")
```

- [ ] **Step 3: 修改 else（非 IDLE）分支，排除 RIGHT_TRACKING 和 LEFT_TRACKING**

关键：`CORNER`、`CROSS`、`TURNING` 状态仍然需要执行图像处理，不能被阻断。因此守卫条件应为排除新状态，而非仅允许 STRAIGHT_TRACKING。

在 else 分支入口添加状态排除检查，结构变为：
```python
else:
    if state not in (ProcessState.RIGHT_TRACKING, ProcessState.LEFT_TRACKING):
        # STRAIGHT_TRACKING 超时检查
        if state == ProcessState.STRAIGHT_TRACKING and t0 is not None:
            if time.time() - t0 >= corner_delay_s:
                state = ProcessState.CORNER
                logging.info(f"State: STRAIGHT_TRACKING -> CORNER (after {corner_delay_s}s)")

        # 保存当前帧到imgprocess
        imgprocess.frame = frame

        # ... 原有的全部图像处理逻辑（preprocess, get_side_line, fit_polynomial, draw_line, imshow 等）保持不变 ...
        # 包括 CORNER -> CROSS -> TURNING 的状态转移逻辑也保持不变

    # RIGHT_TRACKING 和 LEFT_TRACKING 暂无处理逻辑
```

- [ ] **Step 4: 提交**

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "feat: main_video() 状态机支持三路信号"
```

---

### Task 3: 修改 main() 状态机

**Files:**
- Modify: `src/vision_line/scripts/image_process.py:1310` (信号变量)
- Modify: `src/vision_line/scripts/image_process.py:1343-1421` (状态转移)

- [ ] **Step 1: 替换信号变量**

将第 1310 行：
```python
command_received = True
```
改为：
```python
straight_received = True
right_received = False
left_received = False
```

- [ ] **Step 2: 修改 IDLE 分支为三路 if-elif**

将 IDLE 分支：
```python
if state == ProcessState.IDLE:
    if command_received:
        state = ProcessState.STRAIGHT_TRACKING
        t0 = time.time()
        logging.info("State: IDLE -> TRACKING (command received)")
```
改为：
```python
if state == ProcessState.IDLE:
    if straight_received:
        state = ProcessState.STRAIGHT_TRACKING
        t0 = time.time()
        logging.info("State: IDLE -> STRAIGHT_TRACKING (straight received)")
    elif right_received:
        state = ProcessState.RIGHT_TRACKING
        logging.info("State: IDLE -> RIGHT_TRACKING (right received)")
    elif left_received:
        state = ProcessState.LEFT_TRACKING
        logging.info("State: IDLE -> LEFT_TRACKING (left received)")
```

- [ ] **Step 3: 修改 else（非 IDLE）分支，排除 RIGHT_TRACKING 和 LEFT_TRACKING**

与 Task 2 Step 3 相同的结构调整：
```python
else:
    if state not in (ProcessState.RIGHT_TRACKING, ProcessState.LEFT_TRACKING):
        # STRAIGHT_TRACKING 超时检查
        if state == ProcessState.STRAIGHT_TRACKING and t0 is not None:
            if time.time() - t0 >= corner_delay_s:
                state = ProcessState.CORNER
                logging.info(f"State: STRAIGHT_TRACKING -> CORNER (after {corner_delay_s}s)")

        # ... 原有的全部图像处理逻辑保持不变 ...

    # RIGHT_TRACKING 和 LEFT_TRACKING 暂无处理逻辑
```

- [ ] **Step 4: 提交**

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "feat: main() 状态机支持三路信号"
```

---

### Task 4: 验证

- [ ] **Step 1: 语法检查**

```bash
cd d:/programs/ucar_ws && python -m py_compile src/vision_line/scripts/image_process.py
```
Expected: 无报错

- [ ] **Step 2: 全局搜索确认无遗漏**

搜索文件中是否还残留 `command_received` 或未重命名的 `TRACKING`。

```bash
grep -n "command_received" src/vision_line/scripts/image_process.py
grep -n "ProcessState\.TRACKING" src/vision_line/scripts/image_process.py | grep -v STRAIGHT_TRACKING
```
Expected: 无匹配
