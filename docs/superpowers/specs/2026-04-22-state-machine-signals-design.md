# 状态机多信号设计

## 概述

修改 `image_process.py` 中的 `ProcessState` 状态机，将单一的 `command_received` 信号替换为三个互斥输入信号（`straight_received`、`right_received`、`left_received`），新增 `RIGHT_TRACKING` 和 `LEFT_TRACKING` 状态，逻辑留空。

## 现状

- 单一信号：`command_received = True`
- 状态链：`IDLE → TRACKING → CORNER → CROSS → TURNING`
- `TRACKING` 状态执行完整图像处理流水线（preprocess、get_side_line、fit_polynomial 等）
- 状态机在 `main_video()` 和 `main()` 中各有一份

## 目标

### 枚举

```python
class ProcessState(Enum):
    IDLE = 0
    STRAIGHT_TRACKING = 1   # 原 TRACKING 重命名
    RIGHT_TRACKING = 5       # 新增
    LEFT_TRACKING = 6        # 新增
    CORNER = 2
    CROSS = 3
    TURNING = 4
```

值为 5、6 跳开现有值，方便后续插入新状态。

### 信号

三个互斥占位变量替换 `command_received`。由于是硬编码常量，同一时刻仅一个为 True，无需运行时互斥校验：

```python
straight_received = True
right_received = False
left_received = False
```

IDLE 分支中按 `straight_received` → `right_received` → `left_received` 顺序用 if-elif 检查。

### 状态转移

```
IDLE → STRAIGHT_TRACKING   (straight_received)
IDLE → RIGHT_TRACKING      (right_received)
IDLE → LEFT_TRACKING       (left_received)

STRAIGHT_TRACKING → CORNER → CROSS → TURNING  (不变)

RIGHT_TRACKING: 终端状态，无退出转移，停留至程序结束
LEFT_TRACKING:  终端状态，无退出转移，停留至程序结束
```

与现有 `TURNING` 状态行为一致（进入后停留至程序结束）。

### 控制流结构

现有代码中 `else`（非 IDLE）分支对所有非 IDLE 状态执行统一预处理。修改后：

- `STRAIGHT_TRACKING`：进入 `else` 分支，执行完整图像处理流水线（不变）
- `RIGHT_TRACKING` / `LEFT_TRACKING`：跳过整个 `else` 分支，不执行图像处理

实现方式：在 `else` 分支入口处检查当前状态是否为 `STRAIGHT_TRACKING`，只有匹配时才执行后续处理。

### 定时器

进入 `STRAIGHT_TRACKING` 时设置 `t0 = time.time()`（不变）。进入 `RIGHT_TRACKING` 或 `LEFT_TRACKING` 时不设置 `t0`，因此不会有基于延时的状态转移。

### 日志

新增状态转移的日志格式：
```
"State: IDLE -> STRAIGHT_TRACKING (straight received)"
"State: IDLE -> RIGHT_TRACKING (right received)"
"State: IDLE -> LEFT_TRACKING (left received)"
```

原有的 `"State: IDLE -> TRACKING"` 日志同步更新为 `STRAIGHT_TRACKING`。

## 改动范围

仅涉及 `src/vision_line/scripts/image_process.py`：
- `ProcessState` 枚举定义
- `main_video()` 状态机部分
- `main()` 状态机部分
