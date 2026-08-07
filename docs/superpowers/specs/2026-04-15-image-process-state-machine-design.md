# 图像处理状态机（指令触发 + 延时拐点检测 + CROSS 状态）

日期：2026-04-15

## 目标
在 [src/vision_line/scripts/image_process.py](src/vision_line/scripts/image_process.py) 中扩展一个状态机，用于视频帧处理流程：

1. 接收到“指令”（用布尔参数模拟）后才开始执行图像处理。
2. 指令刚收到时：调用 `imgprocess.get_side_line_task_2(..., find_corner=False)`（不搜索拐点）。
3. 经过延时 `corner_delay_s` 后进入拐点检测阶段：调用 `get_side_line_task_2(..., find_corner=True)`。
4. 当本帧同时检测到 `left_c/right_c` 且满足触发条件 `left_c.y >= img_h * 0.9` 且 `right_c.y >= img_h * 0.9` 时，进入新状态 `CROSS`：
   - `get_side_line_task_2(..., find_corner=False)`
   - 执行额外流程函数（暂未实现，先搭框架/占位函数）。

> 重要一致性说明：当前 `get_side_line_task_2()` 的行扫描范围由 `down_ratio=0.95` 决定（从 `int(img_h*0.95)` 往上扫描），因此角点 `y` 最大可达 `int(img_h*0.95)`，大于 `img_h*0.9`，所以 `y >= img_h*0.9` 的触发条件在实践中是可达的。如需更”底部触发”，应同步调整扫描下界（`down_ratio`）或阈值比例。
5. 当 `command_received=False` 时，状态机回到 `IDLE`（停止处理）。

> 说明：这里的 `img_h` 以 **实际处理帧（可能被 resize 后）** 的高度为准。

## 约束/非目标
- 本次聚焦在 `image_process.py` 的视频处理 `main()` 状态机框架，以及为状态机提供必要的数据（corner 结果）。
- 不实现 `CROSS` 状态“其他函数”的具体行为，仅提供稳定的占位接口与调用位置。
- ROS 指令订阅/服务机制不在本次范围（仍用布尔变量模拟）。

## 设计

### 新增/变更的数据
在 `ImageProcess` 对象中维护本帧拐点结果（供状态机读取）：
- `self.left_c: tuple[int,int] | None`
- `self.right_c: tuple[int,int] | None`

每帧进入 `get_side_line_task_2()` 时先清空为 `None`，当检测到拐点时写入。

### 参数
- `command_received: bool`
  - 是否已收到指令（模拟触发信号）
  - 默认：`True`
- `corner_delay_s: float`
  - 从“首次收到指令”开始计时，超过该阈值后启用拐点检测
  - 计时口径：**墙钟时间（`time.time()`）**，与视频帧率无关；离线视频快放/慢放时，该延时表示“处理运行了多久”，而不是“视频时间过去多久”。
  - 默认：`2.0`
- `cross_y_ratio: float`
  - CROSS 状态触发阈值比例（y 坐标阈值 = `img_h * cross_y_ratio`）
  - 默认：`0.9`

### 状态定义
- `IDLE`
  - 条件：`command_received == False` 或尚未收到过指令
  - 行为：不调用 `get_side_line_task_2()`（不做边线/中线更新）

- `TRACKING`
  - 条件：`command_received == True` 且 `now - t0 < corner_delay_s`
  - 行为：调用 `get_side_line_task_2(..., find_corner=False)`

- `CORNER`
  - 条件：`command_received == True` 且 `now - t0 >= corner_delay_s`
  - 行为：调用 `get_side_line_task_2(..., find_corner=True)`

- `CROSS`（新增）
  - 条件：在 `CORNER` 期间，满足“CROSS 触发条件”
  - 行为：
    - 调用 `get_side_line_task_2(..., find_corner=False)`（进入该状态后不再搜拐点）
    - 调用 `imgprocess.run_cross_stage(...)`（占位函数）

### CROSS 触发条件（封装为判断函数）
在 `ImageProcess` 中新增方法（只做判断，不修改状态机）：
- `judge_enter_cross_state(img_shape, y_ratio=0.9) -> bool`

规则：
- `self.left_c is not None and self.right_c is not None`
- `self.left_c[1] >= img_h * y_ratio`
- `self.right_c[1] >= img_h * y_ratio`

其中 `img_h = img_shape[0]`。


### 占位接口（CROSS 额外处理框架）
新增方法：
- `run_cross_stage(binary_img, canvas)`

该函数暂不实现具体算法，只做占位（例如 `pass` 或日志），为后续扩展保留落点。

### 状态变量
- `t0: float | None`
  - 在 `IDLE -> TRACKING` 瞬间记录 `time.time()`
  - 在回到 `IDLE` 时清空

### 状态转移
- `IDLE --(command_received=True)--> TRACKING`
  - 动作：`t0 = time.time()`

- `TRACKING --(time.time() - t0 >= corner_delay_s)--> CORNER`

- `CORNER --(imgprocess.judge_enter_cross_state(...) == True)--> CROSS`
  - 触发检查时机：仅在 **本帧** 调用 `get_side_line_task_2(find_corner=True)` 完成之后检查，保证 `left_c/right_c` 属于当前帧。

- `CROSS` 的退出策略（本次定义为“锁存”）：
  - `CROSS --(command_received=False)--> IDLE`
  - 若 `command_received` 持续为 True，则保持在 `CROSS`（不自动退出）；后续如需“自动退出/回退到 CORNER”，再单独扩展条件。

- `TRACKING/CORNER/CROSS --(command_received=False)--> IDLE`
  - 动作：`t0 = None`
  - 注意：本次不强制清空 `prev_*` 等上一帧边线历史

## 落地点（集成位置）
在 `image_process.py` 的 `main()` 视频处理循环中：

1. 每帧读取 `frame` 后，根据 `command_received` 和 `t0` 推导当前 `state`
2. 若 `state != IDLE`：执行 `preprocess()` -> `return_frame()` -> `get_side_line_task_2(find_corner=...)` -> 拟合/绘制/显示
3. 若 `state == CORNER`：先调用 `get_side_line_task_2(find_corner=True)`，再检查 `judge_enter_cross_state(img.shape, 0.9)`；若为真则立即在同一帧内把 `state` 切到 `CROSS` 并调用 `run_cross_stage(...)`（首帧 CROSS 仍以 `find_corner=True` 执行了边线检测，但同时也会触发 `run_cross_stage`）
4. 若 `state == CROSS`：调用 `get_side_line_task_2(find_corner=False)`，然后调用 `run_cross_stage(...)`
5. 若 `state == IDLE`：跳过 `get_side_line_task_2()`（以及依赖它结果的拟合/绘制部分）

## 验收标准
- `command_received=True`：
  - 指令触发后前 `corner_delay_s` 秒：`find_corner=False`
  - 超过 `corner_delay_s` 秒：`find_corner=True`
  - 当满足 `left_c/right_c` 且 `y >= img_h*0.9`：进入 `CROSS`，随后 `find_corner=False` 且会调用 `run_cross_stage(...)`

- `command_received=False`：
  - 不再调用 `get_side_line_task_2()`，状态回到 `IDLE`
  - `prev_*` 等历史边线信息不在本次强制清空范围
