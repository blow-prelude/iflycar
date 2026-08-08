# 转弯后切换巡线侧与无效帧复用消息设计

## 背景

`find_way_ros.cpp` 的 `STRAIGHT_TRACKING` 当前由 `straight_track_side_` 决定转弯前使用哪一侧边线：

- `LEFT_ONLY` 对应 `LEFT_OFFSET`，由左边线加横向偏移生成中线；
- `RIGHT_ONLY` 对应 `RIGHT_OFFSET`，由右边线减横向偏移生成中线；
- `BOTH` 对应 `MID_AVG`，由左右边线平均生成中线。

检测到拐点消失并确认转弯结束后，现有实现会永久改为 `BOTH + MID_AVG`。新的行为要求改为相反单边：初始沿左边巡线，转弯结束后沿右边巡线；初始沿右边巡线，则转弯结束后沿左边巡线。

消息发布方面，`buildVisionLineMsg()` 在拟合点为空、点数不足或目标索引无效时发送 `[0.0, -1.0]`。短暂丢线会因此把无效数据直接交给控制节点。新的行为要求当前帧无法取得目标点时，继续发送本次任务中上一条有效的 `/vision_line` 数据。

## 目标

- 转弯结束的同一帧立即从初始巡线侧切换到相反巡线侧。
- 切换后继续采用单边搜索和与该侧匹配的偏移中线模式，不再固定切成双边平均。
- 当前拟合点不足以取得 `target_index` 时复用上一条有效消息。
- 方向指令变化后清空消息缓存，避免把上一任务的控制误差带入新任务。
- 保持 `/vision_line` 消息格式不变，仍为 `[x_error, y_raw]`。

## 非目标

- 不修改拐点开始、结束的连续帧确认规则。
- 不修改 `ImageProcess` 的边线搜索、补线、拟合和偏移距离算法。
- 不增加消息时间戳、有效标志或新的 ROS 话题。
- 不为缓存设置超时；同一次任务内连续无效帧会持续保持最后有效值。
- 不修改 `find_way.cpp` 和 Python 实现。

## 涉及文件

| 文件 | 改动 |
|---|---|
| `src/vision_line/src/find_way_ros.cpp` | 派生转弯后的相反巡线配置；替换转弯后双边分支；缓存并复用最后有效消息 |

不需要修改 `image_process.h`、`image_process.cpp`、消息定义或 launch 参数。

## 详细设计

### 1. 同时派生转弯前、转弯后的巡线配置

保留 `straight_track_side_` 作为初始巡线侧。新增两个成员，保存转弯结束后的搜索侧和中线模式：

```cpp
SearchSide post_turn_side_ = BOTH;
MidLineMode post_turn_mode_ = MID_AVG;
```

将 `SearchSide` 到 `MidLineMode` 的映射、左右侧反转分别集中为小型辅助函数，避免构造函数和帧循环中出现多份 `switch`：

```cpp
static MidLineMode midLineModeForSide(SearchSide side)
{
    if (side == LEFT_ONLY)
        return LEFT_OFFSET;
    if (side == RIGHT_ONLY)
        return RIGHT_OFFSET;
    return MID_AVG;
}

static SearchSide oppositeSide(SearchSide side)
{
    if (side == LEFT_ONLY)
        return RIGHT_ONLY;
    if (side == RIGHT_ONLY)
        return LEFT_ONLY;
    return BOTH;
}
```

构造时一次性计算：

```cpp
straight_side_ = straight_track_side_;
straight_mode_ = midLineModeForSide(straight_side_);
post_turn_side_ = oppositeSide(straight_side_);
post_turn_mode_ = midLineModeForSide(post_turn_side_);
```

映射关系为：

| 初始配置 | 转弯前 | 转弯结束后 |
|---|---|---|
| `LEFT_ONLY` | `LEFT_ONLY + LEFT_OFFSET` | `RIGHT_ONLY + RIGHT_OFFSET` |
| `RIGHT_ONLY` | `RIGHT_ONLY + RIGHT_OFFSET` | `LEFT_ONLY + LEFT_OFFSET` |
| `BOTH` | `BOTH + MID_AVG` | `BOTH + MID_AVG` |

`BOTH` 没有可定义的“相反侧”，因此保持双边平均。这只是保留现有配置能力；需求的左右切换路径均为单边模式。

不直接修改 `straight_side_`，这样收到新方向指令、`resetCornerTurningState()` 将 `turn_completed_` 复位后，代码会自然恢复初始巡线侧，不需要额外恢复逻辑。

### 2. 转弯结束后使用相反单边

把 `STRAIGHT_TRACKING` 中 `turn_completed_` 分支的双边处理：

```cpp
processor_.set_mid_line_mode(MID_AVG);
processor_.get_side_line_task_2(binary_img, canvas, true, false, BOTH);
processor_.extend_shorter_line_to_match_min_y(proc_w);
```

替换为转弯后配置：

```cpp
processor_.set_mid_line_mode(post_turn_mode_);
processor_.get_side_line_task_2(binary_img, canvas, true, false,
                                post_turn_side_);
if (post_turn_side_ == BOTH)
    processor_.extend_shorter_line_to_match_min_y(proc_w);
```

`find_corner` 保持为 `false`，因为一次任务只判定一次转弯；切换后的单边只负责持续巡线。单边模式不调用 `extend_shorter_line_to_match_min_y()`，该函数只有左右边线同时存在时才有意义。兼容的 `BOTH` 配置继续执行顶部短边补线。

相同替换要应用到“转弯刚在本帧结束”的立即重跑分支。执行顺序为：

```text
本帧按初始侧搜索并检测拐点
  -> updateCornerTurningState() 确认拐点连续消失
  -> turn_completed_ = true，转弯标志清零
  -> 清除并按相反侧立即重跑本帧搜索
  -> 按相反侧计算、拟合和发布中线
```

因此不会多发布一帧初始侧数据。从下一帧开始，入口处的 `turn_completed_` 分支继续使用相反侧。

### 3. 缓存最后一条有效 `/vision_line` 消息

在 `FindWayROS` 中增加只由主循环访问的缓存：

```cpp
std_msgs::Float32MultiArray last_valid_vision_line_msg_;
bool has_last_valid_vision_line_msg_ = false;
```

不需要互斥锁：`buildVisionLineMsg()` 和 `publish()` 都只在 `run()` 所在线程执行，图像与方向回调不访问该缓存。

增加统一的回退函数，保证所有无法生成当前目标点的出口行为一致：

```cpp
std_msgs::Float32MultiArray lastValidOrInvalidMsg() const
{
    if (has_last_valid_vision_line_msg_)
        return last_valid_vision_line_msg_;

    std_msgs::Float32MultiArray msg;
    msg.data = {0.0f, -1.0f};
    return msg;
}
```

`buildVisionLineMsg()` 的处理规则调整为：

1. `line_points` 为空或点数小于 `abs(target_index) + 1`：返回缓存消息；
2. 图像尺寸非法：无法安全缩放，也返回缓存消息；
3. 负索引换算后仍越界：返回缓存消息；
4. 成功取得目标点并生成 `[x_error, y_raw]`：先写入缓存，再返回当前消息。

成功路径示意：

```cpp
msg.data = {static_cast<float>(x_error), static_cast<float>(y_raw)};
last_valid_vision_line_msg_ = msg;
has_last_valid_vision_line_msg_ = true;
return msg;
```

这里缓存最终消息而不是缓存点，原因是上一条消息已经包含当时正确的图像缩放、目标索引和坐标误差，回退时无需用当前帧的尺寸重新解释旧点。

### 4. 缓存生命周期

缓存只在同一次方向任务内有效。在处理 `reset_requested_` 时，与状态机复位一起执行：

```cpp
has_last_valid_vision_line_msg_ = false;
last_valid_vision_line_msg_.data.clear();
```

其行为为：

- 节点启动后尚未生成有效数据：无效帧仍发送 `[0.0, -1.0]`；
- 同一次直行或左右任务中短暂丢点：发送上一条有效消息；
- 直行任务内部从转弯前切到转弯后：不清缓存，切换帧若无足够点会平滑保持切换前最后值，取得相反侧有效点后立即覆盖；
- 收到新的 `straight`、`left`、`right` 或 `stop` 指令：清缓存，新任务不会复用旧任务数据。

不在普通 `IDLE` 循环中反复清空缓存，只在实际收到方向回调并处理复位请求时清空。

## 数据流

```text
初始 LEFT_ONLY
  -> LEFT_ONLY 搜索 + LEFT_OFFSET 中线
  -> 拐点连续出现：turning_active_ = true
  -> 拐点连续消失：turn_completed_ = true
  -> 同帧 RIGHT_ONLY 重跑 + RIGHT_OFFSET 中线
  -> 后续帧持续 RIGHT_ONLY

初始 RIGHT_ONLY
  -> 流程镜像，转弯结束后持续 LEFT_ONLY + LEFT_OFFSET

每次生成发布消息
  -> 当前拟合点足够且索引有效：生成、缓存、发布当前值
  -> 当前拟合点不足：有缓存则发布缓存
                       无缓存则发布 [0.0, -1.0]
```

## 边界条件

| 场景 | 预期行为 |
|---|---|
| 转弯结束确认发生在当前帧 | 当前帧立即按相反侧重跑，只发布相反侧结果 |
| 初始为 `LEFT_ONLY` | 结束后使用 `RIGHT_ONLY + RIGHT_OFFSET` |
| 初始为 `RIGHT_ONLY` | 结束后使用 `LEFT_ONLY + LEFT_OFFSET` |
| 初始为 `BOTH` | 结束前后均为 `BOTH + MID_AVG` |
| 相反侧切换后的第一帧点不足 | 复用切换前最后有效消息；相反侧有效后更新缓存 |
| 连续多帧点不足 | 持续发布同一条缓存消息，不设超时 |
| 节点启动即点不足 | 因无缓存，发布 `[0.0, -1.0]` |
| 方向变化后第一帧点不足 | 缓存已清空，发布 `[0.0, -1.0]` |
| 正索引或负索引换算后越界 | 视为当前帧无有效目标点，复用缓存 |

## 验证方案

### 编译检查

使用工作空间现有 catkin 构建方式编译 `vision_line`，确认新增成员、辅助函数和消息拷贝均能通过编译。

### 巡线侧切换

1. 保持 `straight_track_side_ = LEFT_ONLY`，输入能稳定触发拐点出现、消失的图像序列。
2. 确认转弯前只搜索左边，转弯结束日志出现的同一帧改为只搜索右边。
3. 确认切换后中线满足 `right.x - turning_mid_offset`，后续帧不恢复双边平均。
4. 将初始侧改为 `RIGHT_ONLY`，重复验证镜像行为：结束后只搜索左边，中线满足 `left.x + turning_mid_offset`。
5. 触发方向重置后再次进入 `STRAIGHT_TRACKING`，确认恢复配置的初始侧。

### 消息回退

1. 先输入足够拟合点，记录发布值 `A = [x_error, y_raw]`。
2. 下一帧令拟合点数量小于目标索引要求，确认仍发布 `A`，而不是 `[0, -1]`。
3. 再输入新的有效帧得到 `B`，确认缓存更新并发布 `B`。
4. 连续输入多个无效帧，确认持续发布 `B`。
5. 发送新的方向指令后先输入无效帧，确认缓存已清空并发布 `[0, -1]`；随后输入有效帧，确认重新建立缓存。
6. 分别覆盖 `STRAIGHT_TRACKING` 的 `straight_target_p_index` 和 `RIGHT_TRACKING` / `LEFT_TRACKING` 的 `left_target_p_index`，确认两条发布路径共用相同回退规则。

## 风险与取舍

- 保持最后值能滤掉短暂丢点，但连续长时间丢线时控制节点无法从消息内容区分“新数据”和“保持值”。本次按需求不增加超时或有效标志。
- 转弯结束切换帧若相反侧尚未建立稳定点，会暂时沿用切换前的最后控制值；相比立即发布 `[0, -1]`，这能保持控制连续性，下一条相反侧有效数据会覆盖它。
- `turning_mid_offset` 同时用于左右单边模式，沿用当前镜像语义；本次不拆分左右偏移参数。
- `target_y_`、`turning_target_y_` 和 `turning_end_x_error_abs_max_` 当前未参与这两项逻辑，本次不顺带清理。

## 实施清单

- 在 `FindWayROS` 中增加巡线侧映射和左右反转辅助函数。
- 派生 `post_turn_side_`、`post_turn_mode_`。
- 将转弯完成常驻分支和同帧立即重跑分支改为相反单边配置。
- 增加最后有效消息及其存在标志。
- 让 `buildVisionLineMsg()` 的无效出口统一回退、成功出口更新缓存。
- 在方向状态机复位时清空消息缓存。
- 编译并按左右镜像、转弯同帧切换、首次无缓存、连续无效帧和方向重置场景验收。
