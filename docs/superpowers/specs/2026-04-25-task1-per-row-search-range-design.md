# Task1：get_side_line_task_1 按行递推 + 上窄下宽搜索窗口（设计）

日期：2026-04-25

## 背景
当前 [src/vision_line/scripts/image_process.py](src/vision_line/scripts/image_process.py) 里的 `ImageProcess.get_side_line_task_1()` 在逐行扫描时，会用“上一帧”的边线（`prev_left/right` 通过 `_get_search_start_point()`）来决定本帧每一行的 `search_start`，并使用固定 `self.search_range` 作为水平搜索窗口大小。

本次修改目标是：**去掉上一帧依赖**，改为**同一帧内按行递推**来决定每行搜索起点；同时让搜索窗口在图像上方更窄、下方更宽。

范围：仅修改 `src/vision_line/scripts/image_process.py`（`image_process_ros.py` 暂不改）。

## 目标
- `search_start` 不再依赖上一帧边线，改为同帧“上一行(y更大、更靠下)”的 x 递推。
- 搜索窗口随 y 上移变小（越靠上越窄）。
- 增加最小必要的回退机制，避免某侧连续丢点后搜索漂移。
- 保持现有后处理不变：`_fill_missing_line / _linear_interpolation / _fill_boundary / mid_line`。

## 非目标
- 不改预处理、stopline 检测、task2。
- 不改稳定起始（stable-start）的核心语义。
- 不在本次同步 ROS 脚本实现。

## 方案概述
### 1）按行递推的搜索起点（左右独立）
在逐行扫描（y 从大到小）的循环中维护两侧”上一行参考 x”：

- `prev_row_left_x`：初始为 `mid_x`
- `prev_row_right_x`：初始为 `mid_x`

`search_offset` 保持现有值 `30` 不变。其含义是：**从上一行 x 向内侧（靠近中线方向）偏移**，使搜索起点偏向图像中央，搜索窗口向外展开。这与当前代码的 offset 方向一致。

对每个 y：
- 左线：
  - `search_start_left = clamp(prev_row_left_x + search_offset, 0, w-1)`
  - 搜索窗口：`[search_start_left - range(y), search_start_left)`（从 search_start 向左搜索）
- 右线：
  - `search_start_right = clamp(prev_row_right_x - search_offset, 0, w-1)`
  - 搜索窗口：`[search_start_right, search_start_right + range(y))`（从 search_start 向右搜索）

若本行找到候选点并且 `_add_point_with_stable_start(...)` 返回 `True`（该点被正式加入边线列表），则更新：
- `prev_row_left_x = x`（左）
- `prev_row_right_x = x`（右）

这包括稳定缓冲区刷入的那一行（`_add_point_with_stable_start` 在 `stable_buf` 满 `init_stable_count` 个点后返回 `True`，此时最后一个点即为首次更新的 `prev_row_*_x`）。

如果本行未成功加入，则对应 `prev_row_*_x` 保持不变（继续沿用上一行的值）。

### 1.1）需要删除的死代码
以下变量和分支在 `get_side_line_task_1()` 中将被移除：
- `use_prev_supple`、`prev_left`、`prev_right` 变量声明（当前行 634-644）
- `is_first_frame` 标志及其分支（当前行 646-652）
- 对 `_get_search_start_point()` 的两次调用（当前行 675-677、705-707）
- 基于上述变量的 `search_start` 赋值分支（当前行 679、709）

### 2）搜索窗口大小：二段阶梯函数（按 y_norm 分档）
定义 `y_norm = y / img_h`。扫描区间为 `y_norm ∈ [0.55, 0.90]`，按以下**二段阶梯**分配窗口宽度：

- 若 `y_norm > 0.6`：`range = 50`（底部 ~67% 的扫描区域，窗口较宽）
- 否则：`range = 30`（顶部 ~33% 的扫描区域，窗口较窄）

这是**分段常量**（非渐变），在 `y_norm = 0.6` 处有一个跳变。

### 3）回退策略（只在 stable 后计数）
由于按行递推会“沿用上一行 x”，如果某侧连续多个 y 没有稳定加入点，可能出现搜索漂移。为降低风险，增加轻量回退：

- 为左右各维护 `miss_count`。
- **只有在该侧进入稳定阶段后（`stable[0] == True`）才开始计数 miss**，避免初始稳定缓冲期误触发。

对某侧在一行 y 的“miss”定义：
- 窗口内无 `candidates`，或
- 有 `candidates` 但 `_add_point_with_stable_start(...)` 返回 `False`（未能稳定加入）。

当 `miss_count >= N` 时触发回退：
- 将该侧 `prev_row_*_x` 重置为 `mid_x`
- 将该侧 `miss_count` 清零

选择 `mid_x` 而非"最后已知好点"的原因：连续 miss 多行后，"最后已知好点"可能已经漂移到错误位置，回到 `mid_x` 更安全。

参数：`N = 3`。

## 验收标准
- `get_side_line_task_1()` 的每行搜索窗口不再使用 `prev_left/right` 或 `_get_search_start_point()` 做引导。
- `get_side_line_task_1()` 内与上一帧搜索相关的变量（`use_prev_supple`、`prev_left/right`、`is_first_frame`）及对 `_get_search_start_point()` 的调用被删除。
- 在常见场景下边线点仍可稳定生成，并且后处理生成的 `supple_left/right_line` 与 `mid_line` 保持对齐特性不被破坏。

## 测试建议
- 运行现有 pytest：特别是 `src/vision_line/scripts/test_fill_missing_line.py`（确保补线/对齐逻辑不受影响）。
- **使用虚拟环境测试**：请使用解释器路径 `"D:\Anaconda\envs\opencv39\python.exe"` 运行测试，以确保依赖版本一致。例如：
  - `"D:\Anaconda\envs\opencv39\python.exe" -m pytest src/vision_line/scripts/test_fill_missing_line.py -q`
- 手动运行视频/相机入口，进入 `RIGHT_TRACKING/LEFT_TRACKING`（会走 task1），观察可视化点集是否稳定、顶部缩窗是否导致过度丢线。

## 兼容性与后续
- 本次修改仅限 task1，易回滚。
- 若效果稳定，再将相同逻辑移植到 `src/vision_line/scripts/image_process_ros.py`。
