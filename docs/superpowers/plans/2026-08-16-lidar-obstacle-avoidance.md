# 巡线雷达避障设计方案

## 1. 目标

在视觉巡线节点运行于 `STRAIGHT_TRACKING`、`LEFT_TRACKING` 或
`RIGHT_TRACKING` 时订阅二维雷达。如果雷达在车体正前方的检测区域内发现近距离
障碍物，则临时接管底盘，按以下固定动作绕开障碍：

```text
沿选定方向横移 → 向前移动 → 沿相反方向横移回原巡线路径
```

动作完成后清除停止线计数和停止线去抖子状态机，丢弃避障前的视觉线缓存，按
当前方向重新进入巡线。

本方案只设计上述固定避障流程，不引入局部规划、障碍物跟踪、绕行距离自适应或
`BOTH` 搜索侧支持。

## 2. 已确认的现状

- 雷达发布 `/scan`，消息类型为 `sensor_msgs/LaserScan`。
- 底盘接收 `/cmd_vel`，消息类型为 `geometry_msgs/Twist`；车体坐标约定为
  `linear.x > 0` 向前、`linear.y > 0` 向左、`linear.y < 0` 向右。
- `find_way_ros` 只发布 `/vision_line`，实际的巡线控制节点
  `vision_line_node_fixed` 根据该消息持续向 `/cmd_vel` 发布速度。
- `/start_vision1` 已是两个节点之间的巡线控制开关：图像节点进入跟踪态时置
  `1`，控制节点读到 `0` 时停止产生新的巡线速度命令。
- `resetStopLineState()` 已能一次清除 `stop_line_count_`、`stop_phase_`、
  `far_run_`、`near_run_`、`miss_run_`、`in_stop_` 和 STOP 退出计时状态。
- `src/vision_line/src/find_way_final_ros .cpp` 是 final 流程的独立源码，不与
  `src/vision_line/src/find_way_ros.cpp` 同步修改。final 流程预期把它构建为
  `find_way_ros_final`，由 `start_final_all.launch` 启动，并由
  `managed_node_client_final.py` 管理生命周期。
- 当前 final 接线尚未完成：`start_final_all.launch` 已引用
  `find_way_ros_final`，但 `vision_line/CMakeLists.txt` 尚未声明该目标；launch
  仍启动普通 `managed_nodes_client.py`，而 `managed_node_client_final.py` 当前为
  空文件且尚未安装。上述接线属于本方案实施范围。

## 3. 关键设计决定

### 3.1 实施文件

避障实现只落在 `src/vision_line/src/find_way_final_ros .cpp`，不修改普通流程使用的
`src/vision_line/src/find_way_ros.cpp`。CMake 新增独立可执行目标
`find_way_ros_final`，并为它配置与普通节点相同的 include、`image_process`、
catkin 和 `ground_perspective` 链接依赖。

当前源码文件名在 `.cpp` 前包含一个空格，CMake 源文件路径必须加引号。本需求不
顺带重命名该文件，避免把文件整理和功能改动混在一起。

### 3.2 雷达判定

新增 `/scan` 订阅者。每次回调只做一次轻量判定并保存结果：

1. 遍历车头正前方 `[-检测半角, +检测半角]` 内的雷达点。
2. 忽略 `NaN`、`Inf`、小于 `range_min` 或大于 `range_max` 的数据。
3. 取其余点的最小距离；最小值小于等于触发距离时记为“前方有障碍”。
4. 雷达尚未收到有效扫描时不触发避障，继续原巡线行为。

检测半角和触发距离按“写死动作”的要求放在 cpp 匿名命名空间常量中，不新增
动态参数。具体数值见第 8 节待确认项。

只在以下条件同时成立时启动避障：

- 节点 `enabled_ == true`；
- 主状态为三个巡线状态之一；
- 当前不在 STOP；
- 没有正在执行的避障动作；
- 避障触发器已武装。

一次避障结束后先解除武装。只有后续收到一帧明确“前方无障碍”的新雷达扫描，
才重新武装，以免避障完成瞬间使用旧扫描结果重复触发，同时仍允许后续避让新的
障碍物。

### 3.3 避障方向映射

触发瞬间根据 `state_` 和 `straight_track_side_` 计算一次方向并保存，动作过程中
不再修改：

| 触发时巡线状态 | 横移方向 | `/cmd_vel.linear.y` 符号 |
| --- | --- | ---: |
| `LEFT_TRACKING` | 右 | 负 |
| `RIGHT_TRACKING` | 左 | 正 |
| `STRAIGHT_TRACKING` 且 `straight_track_side_ == LEFT_ONLY` | 左 | 正 |
| `STRAIGHT_TRACKING` 且 `straight_track_side_ == RIGHT_ONLY` | 右 | 负 |

这张表按需求文字理解为 `LEFT → right`、`RIGHT → left`。`BOTH` 明确不在需求
范围内；现有代码的 `straight_track_side_` 为 `LEFT_ONLY`。若实施后代码意外
进入 `BOTH`，则记录错误并拒绝启动动作，不能默认为任一方向。

### 3.4 非阻塞避障子状态机

在 `FindWayROS` 内新增独立于巡线 `State` 的 `AvoidanceState`：

```text
IDLE
  └─ 前方障碍触发
       ↓
LATERAL_OUT       linear.y = direction_sign × lateral_speed
       ↓ 达到 lateral_distance / lateral_speed
FORWARD           linear.x = forward_speed
       ↓ 达到 forward_distance / forward_speed
LATERAL_BACK      linear.y = -direction_sign × lateral_speed
       ↓ 达到 lateral_distance / lateral_speed
IDLE + 重新巡线
```

每次主循环根据 `std::chrono::steady_clock` 计算阶段经过时间，并以现有
`loop_rate_` 重复发布本阶段 `Twist`。禁止使用阻塞的 `sleep` 或内层 `while`，
以保证雷达、方向和启停服务回调始终能被处理。

横移和前进距离是由“固定速度 × 固定时长”得到的名义距离，不使用 `/odom`
闭环。这是满足“写死动作”的最小实现，但实际距离会受地面、轮胎打滑和底盘速度
误差影响。如果验收要求真实位移达到指定米数，应另行改成里程计闭环方案。

每一阶段只设置本阶段使用的速度分量，其余 `Twist` 分量保持为零；阶段完成、
避障取消及节点退出时都至少发布一次全零 `Twist`。

### 3.5 `/cmd_vel` 控制权切换

final 图像节点和 `vision_line_node_fixed` 不能同时控制 `/cmd_vel`。避障开始时：

1. 将 `/start_vision1` 设为 `0`，使视觉控制节点停止产生新的巡线速度；
2. 停止发布 `/vision_line`；
3. 由 `find_way_ros_final` 按避障状态机发布 `/cmd_vel`。

避障完成后不直接恢复旧的视觉消息，而是先复位巡线数据，再令原主状态机从
`IDLE` 按当前 `direction_` 重新进入对应跟踪态。该既有转移会把
`/start_vision1` 重新设为 `1`，随后恢复 `/vision_line` 发布，控制权回到
`vision_line_node_fixed`。

### 3.6 完成后的复位顺序

`LATERAL_BACK` 完成后按下列顺序处理：

1. 发布全零 `/cmd_vel`；
2. 调用 `resetStopLineState()`，清除停止线计数及完整去抖状态；
3. 调用 `processor_.clear_lines()`，清除避障前的边线/拟合线；
4. 设置 `has_last_valid_msg_ = false`，防止沿用避障前的 `/vision_line`；
5. 设置 `state_ = IDLE`，保留 `direction_`；
6. 下一轮按既有 `IDLE → *_TRACKING` 转移重新巡线。

此流程不修改 `/vision_line_done`。避障只允许在 `in_stop_ == false` 时触发，正常
情况下该参数尚未被置 `1`。

### 3.7 中断行为

- 收到方向 `stop`：立即取消避障、发布零速度，并走现有方向复位逻辑；不再恢复
  巡线。
- 服务将节点设为 disabled：立即取消避障并发布零速度，然后执行现有
  `resetProcessingState()`。
- 避障过程中收到 `left`、`right` 或 `straight`：当前三段动作按触发时保存的
  横移方向完成；新方向由 `direction_` 保存，动作结束后按新方向重新巡线。
- ROS 退出：发布零速度后结束节点。

## 4. 主循环接入位置

主循环调整后的优先级为：

```text
ros::spinOnce
  → disabled 处理
  → stop 指令/避障取消处理
  → 正在避障：更新并发布动作，continue
  → STOP 五秒自动禁用检查
  → 巡线状态下检查雷达触发
      → 若触发：切走巡线控制权并发布第一段动作，continue
  → 原有相机取帧、方向复位、视觉处理和停止线检测
```

避障更新放在相机取帧之前，因此即使暂时没有新相机画面，已经开始的避障动作也
能继续执行和结束。

## 5. 代码改动范围

### `src/vision_line/src/find_way_final_ros .cpp`

- 增加 `sensor_msgs/LaserScan.h`、`geometry_msgs/Twist.h`。
- 增加 `/scan` 订阅和 `/cmd_vel` 发布。
- 增加雷达结果、重新武装标志、避障方向快照和 `AvoidanceState` 成员。
- 增加雷达回调、方向映射、启动/更新/完成/取消避障的私有函数。
- 在主循环中按第 4 节顺序接入避障状态机。
- 在 disabled、`stop` 和 ROS 退出路径保证发布零速度。
- 不改图像处理算法、停止线检测算法和现有巡线 PID。

### `src/vision_line/CMakeLists.txt`

- 在 catkin 组件中显式增加 `sensor_msgs`。
- 新增 `find_way_ros_final` 可执行目标，源文件使用带引号的
  `"src/find_way_final_ros .cpp"`。
- 为新目标配置与 `find_way_ros` 相同的 include 目录和链接库；普通目标保持
  不变。

### `src/vision_line/package.xml`

- 增加 `sensor_msgs` 的 build、build_export 和 exec 依赖。

### `src/startup_scripts/launch/start_final_all.launch`

- 保持 final 图像节点为 `type="find_way_ros_final"`、`name="image_process"`。
- 将生命周期管理节点的 `type` 改为 `managed_node_client_final.py`。
- final 流程仍使用 `/image_process/set_enabled` 和
  `/vision_line_node/set_enabled`；因为普通图像节点不会同时在该 launch 中启动，
  不会产生服务名冲突。

### `src/startup_scripts/scripts/managed_node_client_final.py`

- 实现与现有管理器相同的 `SetBool` 生命周期协议和参数轮询行为。
- 管理 final launch 中的 `traffic_light_ros`、final `image_process`、
  `vision_line_node` 和 `find_signal`。

### `src/startup_scripts/CMakeLists.txt`

- 将 `managed_node_client_final.py` 加入 `catkin_install_python`，确保安装空间下
  roslaunch 也能找到它。

### `docs/guides.md`

- 实现完成后补充 `/scan`、`/cmd_vel`、避障方向表、动作状态机和日志说明。

本需求不修改 `find_way_ros.cpp` 或 `find_way.cpp`：前者属于普通流程，后者是无
ROS 雷达和底盘话题的离线入口。`/scan` 和 `/cmd_vel` 继续使用现有全局话题，
不新增 launch 参数。

## 6. 实施任务

### Task 1：加入雷达检测与方向映射

- [ ] 声明消息依赖、subscriber/publisher 和固定常量。
- [ ] 实现有效雷达点过滤及前方最小距离判定。
- [ ] 实现第 3.3 节方向映射，并在触发日志中打印状态、横移方向和最近距离。
- [ ] 验证无有效雷达点、范围外点不会触发。

### Task 2：实现非阻塞三段动作

- [ ] 加入 `AvoidanceState`、阶段起始时间和方向快照。
- [ ] 实现 `LATERAL_OUT → FORWARD → LATERAL_BACK` 的时间转移。
- [ ] 在整个动作期间关闭视觉控制权和 `/vision_line` 发布。
- [ ] 验证各阶段的 `Twist` 只有预期分量非零，结束时为全零。

### Task 3：复位与中断

- [ ] 动作完成后清空停止线、视觉线缓存并从 `IDLE` 重新巡线。
- [ ] 实现“收到一帧无障碍扫描后才重新武装”。
- [ ] 将 `stop`、disabled 和 ROS 退出接到统一取消函数。
- [ ] 验证中断后没有任何非零速度继续发布。

### Task 4：构建与现场验证

- [ ] 补齐 `find_way_ros_final` 构建目标、final 管理脚本及 final launch 接线。
- [ ] 更新两个相关 `CMakeLists.txt`、`package.xml` 和 `docs/guides.md`。
- [ ] 在目标 ROS Noetic 环境构建 `vision_line`。
- [ ] 通过 `start_final_all.launch` 确认启动的是 final 可执行文件和 final 管理器，
      并验证 `/image_process/set_enabled` 可正常启停 final 节点。
- [ ] 先架空车轮或低速空场测试速度符号，再放置软质障碍物做全流程测试。

## 7. 验收标准

1. 无障碍或障碍距离大于阈值时，三个巡线状态的行为与修改前一致。
2. `LEFT_TRACKING` 触发后依次观察到右移、前进、左移；`RIGHT_TRACKING` 为
   左移、前进、右移。
3. `STRAIGHT_TRACKING` 在 `LEFT_ONLY` 时依次左移、前进、右移；切为
   `RIGHT_ONLY` 时方向完全相反。
4. 避障期间 `/vision_line` 不再发布，`/start_vision1 == 0`，且 `/cmd_vel`
   只有避障节点在产生有效运动命令。
5. 动作完成后首条巡线消息来自新识别的线，不复用避障前缓存；日志显示
   `stop_line_count_ == 0`、`stop_phase_ == STOP_IDLE`，随后恢复原方向巡线。
6. 动作后障碍仍在前方时不会立即重复触发；雷达先报告一次无障碍后，新的障碍
   可以再次触发。
7. 任意动作阶段收到 `stop` 或 disabled，底盘立即收到全零 `/cmd_vel`，避障不
   再继续。
8. `catkin_make --pkg vision_line startup_scripts`（或项目等价构建命令）成功；
   `start_final_all.launch` 可找到 `find_way_ros_final` 和
   `managed_node_client_final.py`，且普通 `find_way_ros` 行为不变。

## 8. 已确认语义与待确定数值

以下语义已由用户确认：

- `LEFT_TRACKING → 右移`，`RIGHT_TRACKING → 左移`；STRAIGHT 按
  `straight_track_side_` 选择横移方向。
- “移动一定距离”接受固定速度乘固定时长得到的开环名义距离，不接入 `/odom`
  闭环。

以下内容需求中没有数值，设计不擅自填写：

| 项目 | 待确认值 |
| --- | --- |
| 雷达前方检测半角 | `TBD`（度） |
| 障碍触发距离 | `TBD`（米） |
| 横移距离 | `TBD`（米） |
| 横移速度 | `TBD`（米/秒） |
| 前进距离 | `TBD`（米） |
| 前进速度 | `TBD`（米/秒） |

这些数值确认后，实施阶段只需填写常量并按上述任务执行，无需重新设计结构。
