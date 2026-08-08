# 用水平白线判定转弯结束并切换巡线侧设计（find_way.cpp）

## 背景

`find_way.cpp` 的 `STRAIGHT_TRACKING` 状态当前用**拐点**驱动一次转弯的生命周期：

1. `cornerDetected()` 读取 `get_left_corners()` / `get_right_corners()`；
2. `updateCornerTurningState()` 用连续帧确认——拐点连续出现 `corner_confirm_frames` 帧 → `turning_active`；拐点再连续消失 `turning_end_confirm_frames` 帧 → `turn_completed`；
3. `turn_completed` 后，`get_side_line_task_2(...)` 切到 `post_turn_side`（相反单边）巡线。

`get_side_line_task_2()` 中 `find_corner` 参数控制是否在边线搜索时顺带检测拐点。当前转弯未结束时传 `true`，结束后传 `false`。

新的需求改为：**不再用拐点判断转弯开始与结束，改用水平白线 `ImageProcess::get_stop_line`**；并且**不判断转弯开始**——初始状态即视为“转弯进行中”；当水平白线中点的 y 值大于阈值时认为转弯结束，切换到另一侧巡线。

`ImageProcess::get_stop_line(binary, canvas, is_draw)` 已实现，返回 `std::vector<int>{x_center, y}`（无检测时返回空 `{}`）。其检测 ROI 由 `ImageProcessConfig` 的归一化参数决定，默认：

- `stop_roi_y0 = 0.65`、`stop_roi_y1 = 0.85`（图像下方 65%–85% 的横带）
- `stop_roi_x0 = 0.30`、`stop_roi_x1 = 0.70`
- `stop_kernel_w = 15`（横向形态学核，削弱斜线）、`stop_min_width = 80`（最短宽度）

`STRAIGHT_TRACKING` 中的 `binary_img` 是经透视变换后的 320×240 二值图，因此 `get_stop_line` 返回的 y 是该图坐标系下的绝对像素，取值大致落在 `[0.65·240, 0.85·240] = [156, 204]` 区间内。

## 目标

- 删除拐点驱动的转弯判定（`cornerDetected`、`CornerTurningState`、`updateCornerTurningState`），不再检测、不再使用拐点。
- 不判定转弯开始：进入 `STRAIGHT_TRACKING` 即视为转弯进行中（`turn_completed` 初始为 `false`）。
- 每帧调用 `get_stop_line`，当水平白线中点 y 超过阈值（连续 `turning_end_confirm_frames` 帧）时置 `turn_completed = true`，并在**同一帧**立即切到相反侧 `post_turn_side` 巡线，避免中线切换延迟到下一帧。
- `get_side_line_task_2()` 的 `find_corner` 两个分支均改为 `false`。
- 阈值与确认帧数可配，便于实车标定。

## 非目标

- 不修改 `ImageProcess`（`image_process.h` / `image_process.cpp`），仅复用现成的 `get_stop_line`。
- 不改动 `RIGHT_TRACKING` / `LEFT_TURNING`（`find_way_ros.cpp`）和 `RIGHT_TURNING` / `LEFT_TURNING`（`find_way.cpp`）及其 `get_side_line_task_1` 分支。
- **不修改 `get_stop_line` 的 ROI 参数**（`stop_roi_y0/y1/x0/x1`、`stop_kernel_w`、`stop_min_width` 保持默认）；结束阈值只通过新增的归一化阈值调节。
- 不清理 `corner_delay_s_`、`turning_end_x_error_abs_max_`、`miss_line` 等当前未参与逻辑的既有变量。
- 不增加可视化窗口或新的消息字段。

## 同步说明

记忆中约定：`find_way_ros.cpp` 与 `find_way.cpp` 的巡线逻辑必须同步。本次**同时改动两个入口**，二者均由“拐点驱动”改为“水平白线驱动”，`STRAIGHT_TRACKING` 分支语义保持一致。`find_way_ros.cpp` 额外维护对外转弯标志 `turning_flag_param_`（默认 `/start_vision_line2`），其置位时机随新模型调整（见“详细设计 §5”）。

## 涉及文件

| 文件 | 改动 |
|---|---|
| `src/vision_line/src/find_way.cpp` | 删除拐点判定相关函数与状态；新增水平白线阈值参数与结束确认状态；改写 `STRAIGHT_TRACKING` 分支；转弯结束判定封装为 `checkStopLineTurnEnd` |
| `src/vision_line/src/find_way_ros.cpp` | 同步删除拐点判定；新增 `stop_line_end_y_thresh_` 参数与结束确认状态；改写 `STRAIGHT_TRACKING` 分支；调整 `turning_flag_param_` 置位时机；判定封装为 `checkStopLineTurnEnd` 成员方法 |
| `src/vision_line/include/image_process.h` | `get_side_line_task_2` 新增 `anchor_side` 参数 |
| `src/vision_line/src/image_process.cpp` | `get_side_line_task_2` 不稳定搜索窗口支持按 `anchor_side` 锚定起点 |

## 详细设计

### 1. 删除拐点判定

移除以下符号（当前文件 92–157 行）：

- 自由函数 `cornerDetected(ImageProcess &, SearchSide)`
- 结构体 `CornerTurningState`
- 自由函数 `updateCornerTurningState(...)`

`main()` 中同步删除：

- `int corner_confirm_frames = 3;`（不再需要开始确认）
- `CornerTurningState corner_turning;`

`turning_end_confirm_frames` 保留，语义改为“水平白线 y 超过阈值的连续确认帧数”。

### 2. 新增参数与状态

`main()` 参数区新增水平白线结束阈值，沿用归一化 y（与 `ImageProcessConfig::turning_enter_y_thresh` 风格一致，分辨率无关）：

```cpp
float stop_line_end_y_thresh = 0.80f; // 水平白线中点归一化 y 超过该值视为转弯结束
```

转弯状态精简为两个字段（取代原 `CornerTurningState`）：

```cpp
bool turn_completed = false; // 初始即视为转弯进行中，仅判定结束
int stop_line_end_count = 0; // 水平白线 y 超过阈值的连续帧数
```

> 与原始描述“y 值大于一定值”对应：判定式为 `stop_line[1] > stop_line_end_y_thresh * img_h`，等价于 `y_norm > stop_line_end_y_thresh`。用归一化形式是为了在 320×240 处理图上不写死像素。

### 3. 改写 STRAIGHT_TRACKING 分支

**改前**（274–310 行核心）：

```cpp
if (corner_turning.turn_completed)
{
    img_process.set_mid_line_mode(post_turn_mode);
    img_process.get_side_line_task_2(binary_img, canvas, true, false, post_turn_side);
}
else
{
    img_process.set_mid_line_mode(straight_mode);
    img_process.get_side_line_task_2(binary_img, canvas, true, true, straight_side); // find_corner=true
    updateCornerTurningState(corner_turning, cornerDetected(img_process, straight_side),
                             corner_confirm_frames, turning_end_confirm_frames);
    if (corner_turning.turn_completed)
    {
        img_process.set_mid_line_mode(post_turn_mode);
        img_process.get_side_line_task_2(binary_img, canvas, true, false, post_turn_side);
    }
}
```

**改后**：

```cpp
if (turn_completed)
{
    // 转弯结束后常驻相反单边
    img_process.set_mid_line_mode(post_turn_mode);
    img_process.get_side_line_task_2(binary_img, canvas, true, false, post_turn_side);
}
else
{
    // 转弯进行中：初始侧巡线，不再检测拐点
    img_process.set_mid_line_mode(straight_mode);
    img_process.get_side_line_task_2(binary_img, canvas, true, false, straight_side); // find_corner=false

    // 用水平白线判定转弯是否结束
    std::vector<int> stop_line = img_process.get_stop_line(binary_img, canvas, true);
    bool stop_line_low = false;
    if (!stop_line.empty() && binary_img.rows > 0)
    {
        const float y_norm = stop_line[1] / static_cast<float>(binary_img.rows);
        stop_line_low = y_norm > stop_line_end_y_thresh;
    }

    if (stop_line_low)
    {
        if (stop_line_end_count < turning_end_confirm_frames)
            ++stop_line_end_count;
    }
    else
    {
        stop_line_end_count = 0;
    }

    if (stop_line_end_count >= turning_end_confirm_frames)
    {
        turn_completed = true;
        stop_line_end_count = 0;
        std::cout << "Stop line midpoint y_norm > " << stop_line_end_y_thresh
                  << " for " << turning_end_confirm_frames
                  << " consecutive frames; turning finished" << std::endl;

        // 判定结束的这一帧已按初始侧搜索过，立即重跑相反侧，
        // 避免中线切换延迟到下一帧。
        img_process.set_mid_line_mode(post_turn_mode);
        img_process.get_side_line_task_2(binary_img, canvas, true, false, post_turn_side);
    }
}
```

后续 `calculate_mid_line(binary_img, false)`、`fit_polynomial()`、`draw_line(...)` 与窗口显示保持不变。

### 4. 关于 ROI 与阈值的相互作用（标定要点）

按确认意见，**不修改 `get_stop_line` 的 ROI 参数**。其检测 ROI 默认仅覆盖 `y_norm ∈ [0.65, 0.85]`，水平白线随车前进从上往下移动时：

- 进入 ROI（`y_norm ≈ 0.65`）才开始可检测；
- 移出 ROI（`y_norm > 0.85`）后返回空 → 计数被清零。

因此“y 大于阈值”的有效触发窗是 `[stop_line_end_y_thresh, 0.85]`。若阈值取 0.80，触发窗仅 `[0.80, 0.85]`，线段一旦继续下移就会离开 ROI、计数清零，可能凑不够 `turning_end_confirm_frames`。由于 ROI 不动，唯一的调节手段是**下调归一化阈值**（例如 `stop_line_end_y_thresh = 0.72`，把触发窗扩到 `[0.72, 0.85]`），或减小 `turning_end_confirm_frames`。默认值定为 `0.80`，留作实车标定起点；本设计把它设为可配参数，不在文档里写死最优值。

### 5. ROS 节点对外转弯标志（仅 `find_way_ros.cpp`）

`turning_flag_param_`（默认 `/start_vision_line2`）原在拐点确认开始时置 1、拐点确认结束时置 0。新模型没有“转弯开始”事件——进入 `STRAIGHT_TRACKING` 即视为转弯进行中，因此置位时机调整为：

- `IDLE -> STRAIGHT_TRACKING` 转换时置 1（与 `ros::param::set("/start_vision1", 1)` 同处）；
- 水平白线连续达标、`turn_completed_ = true` 时置 0；
- 方向重置（`resetTurningState()`）与节点退出时置 0。

`RIGHT_TRACKING` / `LEFT_TRACKING` 不接触该标志，保持原行为。这样对外仍维持“标志为 1 = 转弯进行中，标志为 0 = 转弯结束”的语义，只是 1 的起点从“拐点确认”提前到“进入直行巡线”。

### 6. 转弯后切侧首帧的搜索起点沿用（`image_process.cpp`）

转弯结束从巡左线切到巡右线时，原 `RIGHT_ONLY` 不稳定搜索窗口固定为 `[mid_x + search_offset, max_edge_x]`，每帧从“中线偏右”重新找线。但刚切侧时目标线仍在画面中部偏左（原左线位置），固定右窗口会漏检。需求是：**仅切侧后首帧**沿用上一帧（巡左线时）的边线位置作为搜索起点，终点向右偏移；之后恢复常规跟踪。

新增 `get_side_line_task_2(..., SearchSide anchor_side = BOTH)` 参数。当 `anchor_side` 与 `side` 相反、且本帧活动侧的上一帧边线为空（说明刚切侧、上一帧还在巡另一侧）时，仅此首帧做种子：

| 切换方向 | side | anchor_side | 首帧种子来源 | 首行窗口 | 之后 |
|---|---|---|---|---|---|
| 左 → 右 | `RIGHT_ONLY` | `LEFT_ONLY` | `prev_left_line_` 最下点 x | `[seed, seed + cur_range]`（向右偏移） | `prev_row_right_x ± cur_range` |
| 右 → 左 | `LEFT_ONLY` | `RIGHT_ONLY` | `prev_right_line_` 最下点 x | `[seed - cur_range, seed]`（向左偏移） | `prev_row_left_x ± cur_range` |
| 其它 / 非首帧 | 任意 | `BOTH`/同侧 | — | 不变（`mid_x ± offset` 等） | 不变 |

判定“首帧”的依据是活动侧 `prev_*_line_` 为空：切侧前一帧在巡另一侧，活动侧未采集，故其上一帧边线为空；从切侧后第二帧起该侧已有上一帧数据，不再沿用对方位置、回到常规搜索。首帧内：底行用上表的“首行窗口”一次，命中后 `prev_row_*_x` 更新为当前行，其上行及后续帧均按 `prev_row_*_x ± cur_range`（后续帧 `prev_row_*_x` 复位为 `mid_x`，即围绕中线的 `± cur_range` 窗口）。命中稳定点进入稳定分支后同样按 `prev_row_*_x ± cur_range`。调用方在两个 post-turn 调用处传入 `straight_side(_)` 作锚定侧，转弯中巡 straight_side 的调用保持默认 `BOTH`。

> 注：当前实际配置 `straight_track_side = LEFT_ONLY`，只会发生“左→右”；“右→左”为对称镜像，左右两侧分支结构一致。

### 7. 判定封装与可视化（两入口）

- 转弯结束判定（检测停止线→算 y_norm→连续帧计数→置位）封装为单一方法：`find_way.cpp` 为自由函数 `checkStopLineTurnEnd(...)` + `StopLineTurningState` 结构体；`find_way_ros.cpp` 为成员方法 `checkStopLineTurnEnd(binary_img, canvas)`。返回 `bool` 表示“本帧刚判定结束”，调用方据此同帧切到相反侧。
- 检测到停止线时在 canvas 上额外绘制：绿色阈值参考线 `y = stop_line_end_y_thresh × 图高`、中点黄色环，便于标定阈值。`get_stop_line(is_draw=true)` 自带的蓝框+红点保留。

## 数据流

```text
进入 STRAIGHT_TRACKING（初始 turn_completed=false，视为转弯进行中）
  -> 每帧初始侧 straight_side 搜索 + straight_mode 中线，find_corner=false
  -> get_stop_line 在 ROI 内找水平白线
       无检测 / y_norm <= 阈值：stop_line_end_count 清零，继续初始侧
       y_norm > 阈值：stop_line_end_count++
  -> 连续 turning_end_confirm_frames 帧达标
       -> turn_completed=true，同帧立即按 post_turn_side 重跑
  -> 后续帧常驻 post_turn_side + post_turn_mode
```

## 边界条件

| 场景 | 预期行为 |
|---|---|
| 进入 `STRAIGHT_TRACKING` 首帧 | `turn_completed=false`，按初始侧巡线，不判定开始 |
| 水平白线尚未进入 ROI（远） | `get_stop_line` 返回空，计数清零，继续初始侧 |
| 白线在 ROI 内但 y 未过阈值 | 计数清零，继续初始侧 |
| 白线 y 过阈值但仅 1–2 帧（抖动） | 未达 `turning_end_confirm_frames`，不结束 |
| 白线 y 连续过阈值达标 | 当帧切相反侧，后续常驻相反侧 |
| 初始侧为 `LEFT_ONLY` | 结束后用 `RIGHT_ONLY + RIGHT_OFFSET` |
| 初始侧为 `RIGHT_ONLY` | 结束后用 `LEFT_ONLY + LEFT_OFFSET` |
| 初始侧为 `BOTH` | 结束前后均为 `BOTH + MID_AVG`（无相反侧可定义） |
| 白线下移过快、离开 ROI 后才达标 | 计数被清零，可能漏判 → 用“标定要点”中的下扩 ROI / 下调阈值缓解 |

## 验证方案

### 编译检查

按工作空间现有 catkin 方式编译 `vision_line`，确认删除拐点相关符号、新增参数与改写分支均能通过编译。

### 行为验证

1. 保持 `straight_track_side = LEFT_ONLY`，输入含水平白线的图像序列。
2. 确认转弯前只搜左边、`find_corner=false`（canvas 上不再出现拐点标记）。
3. 确认水平白线进入 ROI 后 `get_stop_line` 在 canvas 上画出蓝框与红点（中点）。
4. 当中点 y 过阈值并连续达标时，日志输出 “turning finished”，**同一帧**改为只搜右边；后续帧持续右边、中线满足 `right.x - turning_mid_offset`。
5. 调整 `stop_line_end_y_thresh`（如 0.72 / 0.80）与 `turning_end_confirm_frames`（1 / 3），确认触发时机随参数变化。
6. 复核“标定要点”：若默认 ROI 下线段离开过快导致漏判，下调 `stop_line_end_y_thresh`（如 0.72）或减小 `turning_end_confirm_frames` 后重新验证。
7. 将初始侧改为 `RIGHT_ONLY`，重复镜像验证：结束后只搜左边。

## 风险与取舍

- **检测窗受限**：默认 ROI `[0.65, 0.85]` 把“y 大于阈值”的有效窗压窄，线段下移后会失检。ROI 按确认不动，只能靠下调阈值或减小确认帧数缓解。
- **误检风险**：转弯过程中被跟踪的边线或弧线在 ROI 内可能形成横向连通块。现有 `stop_kernel_w`（横向形态学开运算削弱斜/竖线）与 `stop_min_width=80` 已做约束，必要时按场地调参。
- **单向一次性**：`turn_completed` 一旦置位不再复位（与原拐点实现一致），`find_way.cpp` 为单次运行的测试入口，无需多段转弯复位；如需多次转弯，需另加复位路径（非目标）。
- **ROS 标志前置**：`turning_flag_param_` 现在在进入 `STRAIGHT_TRACKING` 时就置 1（含其前的直行段），而非等到拐点确认。若下游节点对“纯直行段不应置 1”有依赖，需另行评估；本次按“初始即转弯”的需求语义实现。

## 实施清单

**两个入口共同**

- 删除拐点判定：`cornerDetected`、`updateCornerTurningState`（ROS 版含 `resetCornerTurningState`）、`corner_confirm_frames(_)`；`find_way.cpp` 另删 `CornerTurningState` 与 `corner_turning`。
- 新增参数 `stop_line_end_y_thresh(_)`、状态 `turn_completed(_)` / `stop_line_end_count(_)`。
- 改写 `STRAIGHT_TRACKING` 分支：两处 `find_corner` 改 `false`，接入 `get_stop_line` 与结束确认，达标同帧切相反侧。
- 结束判定封装为 `checkStopLineTurnEnd`（自由函数/成员方法），并绘制阈值线与中点。
- `get_side_line_task_2` 新增 `anchor_side` 参数；两个 post-turn 调用传入 `straight_side(_)` 作锚定侧，实现切侧后的居中搜索窗口（见 §6）。

**仅 `find_way_ros.cpp`**

- `IDLE -> STRAIGHT_TRACKING` 时置 `turning_flag_param_ = 1`；结束确认与复位/退出时置 0。
- `resetCornerTurningState()` 改名并简化为 `resetTurningState()`：复位 `turn_completed_`、`stop_line_end_count_` 并置标志 0。

**验收**

- 编译 `vision_line`，按“行为验证”逐项验收两个入口。
