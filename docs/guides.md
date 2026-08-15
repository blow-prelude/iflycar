# 节点指南

## `traffic_light_ros`(traffic_light_ros.cpp)红绿灯方向识别节点

### 功能概述

订阅相机图像,用 YOLOv8(rknn, 3 线程 NPU 池)检测红绿灯/箭头,结合传统 CV 箭头方向判别做**投票决策**,确认方向后发布一次 `/vision_line_direction`,随后自动停用等待外部重新启用。下游 `vision_line` 根据该方向选择巡线分支。

### 话题与服务

| 方向 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 订阅 | `/ucar_camera/image_raw` | `sensor_msgs/Image` | 相机图像,队列 1 |
| 发布 | `/vision_line_direction` | `std_msgs/String` | `straight` / `left` / `right`,确认后只发一次 |
| 服务 | `~set_enabled` | `std_srvs/SetBool` | 启用/暂停处理(默认 enabled) |

### 处理流水线

```text
image_callback
  ├─ enabled_ 为 false 直接返回
  ├─ cv_bridge 转 BGR8 + clone,存入 pending_frames_(deque)
  ├─ pool_.put(frame)               # 提交 rknnPool(3 个 NPU worker)
  ├─ queued_frames_ > 3 才 pool_.get()   # 流水线预热
  ├─ 从 pending_frames_ 取对应原始帧(结果与源帧配对)
  ├─ discard_results_ > 0 时丢弃(启动丢前 5 帧 / 重新启用丢在途帧)
  ├─ showResult()                   # 每 1 秒节流:打印检测日志 + imshow
  └─ updateDirectionDecision()
       ├─ bestDirection()           # 取本帧 straight/left/right 中置信度最高者 → model_label
       ├─ model_label 为 left/right 时:classifyArrowDirection(源帧, 检测框) → cv_label
       ├─ decision_.processFrame(model_label, cv_label)   # 见下
       └─ 返回非空 → 发布一次 /vision_line_direction
            ├─ decision_.reset()
            └─ enabled_ = false,销毁窗口,静默等待服务重新启用
```

### 方向决策机制(`DirectionDecisionAccumulator`)

常量:投票窗口 `kVoteWindowSize=10`,确认阈值 `kVoteThreshold=8`,兜底帧数上限 `kFallbackFrameLimit=30`,兜底 CV 连续数 `kFallbackCvStreak=2`。

**启动门槛**:决策器只有在某帧 model_label 是有效方向后才开始计帧,之前的帧(纯 stop/无检出)不计入。

**正常投票模式**:

1. 每帧向滑动窗口(容量 8,先进先出)投一票 model_label。
2. 若 model_label 与 cv_label 都是 left/right 且**一致**,额外再投一票(CV 确认票,即一帧最多两票)。
3. 任一方向票数 ≥ 5 → 该方向胜出,发布。

**CV 兜底模式**:处理满 15 帧仍未有方向达到 5 票 → 切换到 CV-only 模式(打 WARN 日志):此后只看 cv_label,连续 3 帧一致且该两帧 model_label 也是 left/right → 发布该方向。兜底模式下不回退到投票模式。

**发布后**:节点自动 `enabled_ = false`(一次红绿灯只报一次),由任务流程通过 `~set_enabled` 服务重新启用;重新启用时会重置决策器并丢弃在途的旧推理结果。

### CV 箭头判别(`classifyArrowDirection`)

对检测框(外扩 padding,按分辨率缩放)做传统图像处理,返回 `left`/`right`/`unknown`:

1. ROI 转 HSV:绿色灯罩掩膜(35-100 色调)+ 亮色箭头掩膜(V≥212),亮色与绿色膨胀支撑区求交,闭运算去噪。
2. 连通域分析,按尺寸(宽 15-90、高 12-90)、面积、填充率、周边绿色得分/密度过滤候选。
3. 取 `绿色得分 × 填充率` 最高的候选,PCA 判主轴必须水平(否则 unknown)。
4. 把候选按宽度均分 8 个竖条,密度峰值条在前 4 条(≤3)判 `left`,后 4 条判 `right`;峰值并列判 unknown。

### 启动丢帧

构造时 `discard_results_ = 2`:启动后最先取回的 2 帧推理结果直接丢弃(不打印、不参与决策)。原因是相机自动曝光未收敛的前几帧是过曝/欠曝噪声,量化模型会在这种帧上输出铺满全图、置信度饱和(0.998)的大量假框。

### 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `~image_topic` | `/ucar_camera/image_raw` | 输入图像话题 |
| `~initial_enabled` | true | 启动时是否立即开始处理 |
| `~model_path` | `<pkg>/models/bestfp.rknn` | YOLOv8 rknn 模型 |

### 日志说明

- 检测日志(`class=... box=...`)每 1 秒最多打一帧的完整列表,启动初期若仍见大量 0.998 假框,说明坏帧超过了丢弃窗口。
- `direction frame=... model=... cv=... votes=... mode=...` 为每秒节流的状态行,可观察投票进度与是否进入兜底。
- `Published traffic-light direction: ...(source=vote|cv-fallback)` 为最终发布日志。

### 启动说明

- 单独运行 `rosrun traffic_light traffic_light_ros` ，默认开启推理
- 可以通过 `rosservice call /traffic_light_ros/set_enabled "data: true"` 重新使能

---
## `find_signal`(rknn_ros.cpp)信号牌识别节点

### 功能概述

订阅相机图像,用 PPOCR(det + rec)做文字检测与识别,识别信号牌上的类别文本(食品/日用品/电子产品),发布检测框和类别结果,供 `ourgoal`(switch2)做仓储任务的目标选择与横向对准。

### 话题与服务

| 方向 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 订阅 | `/ucar_camera/image_raw` | `sensor_msgs/Image` | 相机图像,队列 1 |
| 发布 | `/signal_detection` | `std_msgs/Float32MultiArray` | `[中心x, 中心y, 左边中点x, 右边中点x]`(像素) |
| 发布 | `/signal_class` | `std_msgs/Int32` | `0`=食品 `1`=日用品 `2`=电子产品 |
| 服务 | `~set_enabled` | `std_srvs/SetBool` | 启用/暂停图像处理(默认 disabled) |

### 处理流水线

```text
image_callback
  ├─ enabled_ 为 false 直接返回
  ├─ cv_bridge 转 BGR8 + clone(队列中的帧必须自己持有内存)
  ├─ pool_.put(frame)               # 提交 rknnPool(thread_count 个 NPU worker)
  ├─ pending_count_ < thread_count_ 时先攒帧(流水线预热)
  ├─ pool_.get()                    # 取回一帧推理结果
  ├─ show_result()                  # visualize 时 imshow
  └─ publish_result()
       ├─ get_biggest_result()      # 取面积最大的文本框,无框 → 重置确认状态
       ├─ 发布 /signal_detection(无确认机制,检出即发)
       └─ classfy(text) → update_class_streak(class_id)
```

### 类别确认机制(连续 5 帧防抖)

`/signal_class` 不是单帧结果,而是**连续 `confirm_frames_`(默认 5)帧识别出同一类别后才确认发布**:

1. 每帧对最大框的 OCR 文本跑 `classfy`,得到 `class_id`(匹配失败为 -1)。
2. 严格"连续":出现不同类别、`-1`、或该帧没有检测框,计数立即归零重数。
3. 累计满 5 帧后置 `last_confirmed_class_`,打 `class %d confirmed after %d consecutive frames` 日志。
4. **确认后每帧持续发布**(不是只发一次):消费端 switch2 用 `maximum_detection_age` 按消息时间戳判断新鲜度,只发一次会被判过期。
5. 断连(出现 -1/无框)即清空状态,下次重新数满 5 帧。类别切换(如 0→1)同样要重新数满 5 帧。

未确认期间(前 4 帧)只打节流日志 `OCR text: ..., streak: n/5`,不发 `/signal_class`。

### 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `~det_model_path` | `<pkg>/models/ppocrv4_det.rknn` | 检测模型 |
| `~rec_model_path` | `<pkg>/models/ppocrv4_rec.rknn` | 识别模型 |
| `~thread_count` | 3 | NPU worker 数,上限 3 |
| `~confirm_frames` | 5 | 类别确认所需连续帧数 |
| `~visualize` | true | 是否开 cv 窗口 |

### 启动说明

---
## `find_way_ros`(find_way_ros.cpp)视觉巡线节点

### 功能概述

订阅相机图像与方向指令,驱动状态机(IDLE / STRAIGHT_TRACKING / RIGHT_TRACKING / LEFT_TRACKING)做赛道巡线:二值化 → 边线搜索 → 中线生成 → 多项式拟合,把拟合线上目标点的横向偏差发布到 `/vision_line` 供下游控制。同时用帧间去抖状态机检测停止线,跨越 `stop_line_target` 条后进入 STOP:停止发布、置 `/vision_line_done=1`,5 秒后自动禁用(`enabled_=false`,进程不退出,可再次经 `~set_enabled` 启用)。默认 disabled,由任务流程经 `~set_enabled` 启用。

方向指令语义与 `traffic_light_ros` 的输出一致(straight/left/right/stop)。

### 话题与服务

| 方向 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 订阅 | `ucar_camera/image_raw`(`~image_topic`) | `sensor_msgs/Image` | 相机图像,队列 1,回调只存最新帧 |
| 订阅 | `/vision_line_direction_out`(`~direction_topic`) | `std_msgs/String` | straight/left/right/stop;收到即请求状态机复位 |
| 发布 | `/vision_line`(`~vision_line_topic`) | `std_msgs/Float32MultiArray` | `[x_error, y_raw]`;STOP 期间不发布 |
| 服务 | `~set_enabled` | `std_srvs/SetBool` | 启用/暂停处理(默认 disabled) |

与任务流程交互的全局参数(`ros::param`):

- `/start_vision1`:IDLE → 任一跟踪态时置 1
- `/start_vision_line2`(`~turning_flag_param`):构造时与禁用时(含 STOP 自动禁用)置 0
- `/vision_line_done`:进入 STOP 时置 1

### 处理流水线

```text
run() 主循环 (~loop_rate)
  ├─ !enabled_ → 以 ~disabled_rate 慢速空转(方向回调也会直接返回)
  ├─ STOP 已持续 5 秒 → 自动禁用:enabled_=false 并复位状态(含 turning_flag_param=0),进程保留
  ├─ 取 latest_frame_ clone(回调只覆盖式存最新帧)
  ├─ reset_requested_(收到新方向)→ 回 IDLE,复位停止线状态与上帧消息缓存
  ├─ IDLE:按 direction_ 切入 STRAIGHT/RIGHT/LEFT_TRACKING,置 /start_vision1=1
  └─ 跟踪态:
       ├─ STRAIGHT_TRACKING:
       │    resize 到 320x240 → warpFixedGroundPerspective 固定地面透视变换
       │    (抛异常/空帧 → 回退用缩放帧) → preprocess 二值化
       │    → get_side_line_task_2(单侧搜索, straight_side_=LEFT_ONLY)
       │    → calculate_mid_line(LEFT_OFFSET 模式) → fit_polynomial
       ├─ LEFT/RIGHT_TRACKING:
       │    原始帧直接 preprocess → get_side_line_task_1(动态窗口双边搜索)
       │    → calculate_mid_line(MID_AVG, left_weight) → fit_polynomial2
       │    left_weight:LEFT=0.65 / RIGHT=0.40(左侧线权重)
       ├─ updateStopLineDetection()      停止线去抖(见下)
       ├─ buildVisionLineMsg() → 非 STOP 则发布
       └─ draw_line(状态+FPS) + imshow(perspective/binary/processed_img)
```

处理链抛 OpenCV/标准异常时清空线数据并节流打 ERROR,不退出进程。

STRAIGHT 的搜索侧 `straight_track_side_` 是代码内写死的 `LEFT_ONLY`(非 ROS 参数),并在构造时一次性派生出中线模式 `LEFT_OFFSET`。

### 目标点与消息语义(`buildVisionLineMsg`)

- 追踪点由拟合线上的索引指定,**支持负索引**(从末端倒数):STRAIGHT 用 `straight_target_p_index=-25`,LEFT/RIGHT 用 `left_target_p_index=-15`(见 `image_process.h` 的 `ImageProcessConfig`)。
- 该点从处理分辨率缩放回原图分辨率后发布 `[x_error, y_raw]`,其中 `x_error = x - 原图宽/2`(目标点相对画面中心的横向偏差,像素)。
- 中线无效(为空/索引越界)时**不发哨兵,沿用上一帧有效消息**;只有从未有过有效帧才发 `{0, -1}` 哨兵。换方向、重新启用都会清空该缓存。
- STOP(`in_stop_`)期间巡线照跑、窗口照显,但**不发布**。

### 停止线检测与 STOP(帧间去抖)

`updateStopLineDetection` 跟踪单条停止线"远端 → 近端"的完整跨越,子状态机(`STOP_IDLE` / `STOP_SEEN_FAR`):

1. 检出的停止线中心 y 小于归一化阈值(`~stop_y_cross`,STRAIGHT 用 `~straight_stop_y_cross`)→ 远端;连续 `~stop_far_min_frames` 帧 → 进入 `STOP_SEEN_FAR`。
2. 之后中心 y 超过阈值(近端)连续 `~stop_near_min_frames` 帧 → 视为完整跨越一条,`stop_line_count_+1` 并复位子状态机。
3. 持续丢检 `~stop_miss_min_frames` 帧 → 放弃当前 phase 回 `STOP_IDLE`。
4. `stop_line_count_ ≥ stop_line_target`(默认 2)→ 进入 STOP:置 `/vision_line_done=1`、抑制发布,5 秒后自动禁用(`enabled_=false`)并置 `turning_flag_param=0`;进程不退出,可再次 `~set_enabled` 启用。

停止线只在固定 ROI(y 0.65-0.97、x 0.30-0.70,见 `ImageProcessConfig`)内检测;两个 y_cross 参数必须严格落在 ROI 的 y 范围内(启动时校验,否则节点直接终止)。

### 方向指令与状态复位

- `directionCallback`:去首尾空白 + 转小写,仅接受 straight/right/left/stop,非法值忽略并 WARN;任何合法值都置 `reset_requested_`,主循环里先复位回 IDLE 再按新方向切入。
- `stop` 只复位,不切入任何跟踪态。
- 重新 `~set_enabled` 时(STOP 自动禁用后再启用同理)清帧缓存、复位状态机与停止线计数,但**保留 direction_**:若保留的方向不是 stop,启用后会立即按旧方向切入跟踪。
- 禁用期间方向消息会被丢弃(回调开头判 `enabled_`),需先启用再发方向。

### 参数

| 参数 | 默认(代码 / launch) | 说明 |
|------|------|------|
| `~image_topic` | `ucar_camera/image_raw` | 输入图像话题 |
| `~vision_line_topic` | `/vision_line` | 输出巡线话题 |
| `~direction_topic` | `/vision_line_direction_out` | 方向指令话题 |
| `~turning_flag_param` | `/start_vision_line2` | 禁用时复位的全局参数名 |
| `~initial_direction` | `stop` / launch 为 `left` | 启动时初始方向 |
| `~loop_rate` | 120 / launch 为 60 | 主循环频率 Hz |
| `~disabled_rate` | 10.0 | 禁用时慢速轮询频率 Hz |
| `~left_tracking_left_weight` | 0.65 | LEFT_TRACKING 中线左侧线权重 |
| `~right_tracking_left_weight` | 0.40 | RIGHT_TRACKING 中线左侧线权重 |
| `~stop_line_target` | 2 | 进入 STOP 所需跨越的停止线条数 |
| `~stop_y_cross` | 0.80 | LEFT/RIGHT 远/近端归一化 y 阈值 |
| `~straight_stop_y_cross` | 0.85 | STRAIGHT 远/近端归一化 y 阈值 |
| `~stop_far_min_frames` / `~stop_near_min_frames` / `~stop_miss_min_frames` | 3 / 3 / 3 | 去抖帧数:远端确认 / 近端确认 / 丢线放弃 |
| `~corner_delay_s` / `~turning_end_x_error_abs_max` / `~vision_target_y` / `~turning_target_y` | 1.5 / 15.0 / 400.0 / 360.0 | 仅加载并校验,当前主流程未使用(旧转弯逻辑保留) |

### 日志与调试窗口

- `State: IDLE -> XXX`、`Direction set to: ...`、`State machine reset to IDLE (direction changed)`:状态机流转。
- `Stop line crossed: count=n (target=m)`、`STOP entered: ...`:停止线跨越与 STOP 进入。
- `STOP has been active for 5 seconds; disabling find_way_ros (enabled_=false)`:STOP 满 5 秒自动禁用;此后回到 disabled 慢速空转,方向消息同样被丢弃。
- imshow 窗口:`perspective`(仅 STRAIGHT)/ `binary` / `processed_img`(叠加边线、中线、目标点、状态与 FPS)。

### 启动说明

- `roslaunch vision_line find_way.launch`,常用参数已做成 launch arg。
- 默认 disabled,启用:`rosservice call /find_way_ros/set_enabled "data: true"`。
- 同目录的 `find_way.cpp` 是同一套巡线逻辑的离线(非 ROS)入口,两处巡线逻辑的改动必须保持同步。

---
## `switch_node`(switch_test2.cpp)任务总调度状态机节点

### 功能概述

整场比赛的**总流程调度器**:从语音唤醒后逃出仓储迷宫开始,依次完成二维码物品领取 → AI 任务分配 → 实体/仿真生产区仓储对准停车 → 远程仿真协同 → 抵达巡线起点并联动启动红绿灯/巡线视觉节点 → 等待巡线结束 → 语音播报收尾。所有阶段在 `switch_test2.cpp` 的 10 Hz 状态机主循环里分发,具体逻辑在 `OURSWITCH` 类(实现拆在 4 个 cpp)中。

### 源码组成

| 文件 | 内容 |
|------|------|
| `switch_test2.cpp` | main + 状态机主循环(仅分发,不含逻辑) |
| `switch2_core.cpp` | 构造/订阅回调/PID/坐标工具/`sendPos` 导航封装/`detectGap` 雷达缺口检测(保留未启用) |
| `switch2_initial_tasks.cpp` | `GotoA`(迷宫逃脱)、`GotoB`(二维码扫码)、`XingHuoAI`(读 AI 参数) |
| `switch2_warehouse.cpp` | `GotoC`(生产区目标搜索与停车,最复杂的一个阶段) |
| `switch2_final_tasks.cpp` | `Gazebo`(仿真协同)、`GotoD`(巡线起点)、`vision_line`(收尾) |

### 状态流转总览

```text
构造:等待 move_base(5s)→ 阻塞等 awake2=1(AI.py 录音结束)
  └─ current_state 初始为 GOTOA_

GOTOA_  迷宫逃脱(超声波 P 控制平移 10 段 + 原地调头)
   └→(main 里硬切)GOTOB_
GOTOB_  多点导航 + 原地转扫码,等 AI.py qr_scan_done / task1_all_done
   └→ XingHuoAI_
XingHuoAI_  读 real_*/sim_* 六个任务参数
   └→ GOTOC1_
GOTOC1_  GotoC(1):实体车生产区找目标车间 → 对准停车 → 播报2
   └→ GOTOC2_
GOTOC2_  GotoC(2):仿真车生产区同流程(命中缓存时直接复用停车点)
   └→ Gazebo_
Gazebo_  置 start_gazebo_sim=1,轮询等 gazebo_sim_done=1 → 播报3
   └→ GOTOD_
GOTOD_  导航到固定巡线粗起点(0.4, -3.08, -1.57)→ 置 start_traffic_light_det=1
   └→ VISION_LINE_
VISION_LINE_  轮询等 vision_line_done=1 → 置 start_vision_line=0 → 播报4 → ros::shutdown()
```

### 话题、服务与跨节点参数

| 方向 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 订阅 | `/ultra` | `pcl_work/ultrasound` | 四路超声波;后/右两路在回调里取负存成有符号值 |
| 订阅 | `/odom` | `nav_msgs/Odometry` | 四元数→RPY 提取 yaw(车体航向反馈) |
| 订阅 | `/scan` | `sensor_msgs/LaserScan` | 缺口检测采样(见"保留未启用"节) |
| 订阅 | `/signal_class` | `std_msgs/Int32` | `find_signal` 确认的类别(0 食品/1 日用品/2 电子) |
| 订阅 | `/signal_detection` | `std_msgs/Float32MultiArray` | 检测框 `[中心x, 中心y, 左中点x, 右中点x]`;同时写参数 `center_x` |
| 发布 | `/cmd_vel` | `geometry_msgs/Twist` | 底盘速度(全向:linear.x/y + angular.z) |
| 发布 | `move_base/cancel` | `actionlib_msgs/GoalID` | 超时取消导航目标 |
| 服务客户端 | `/srv_getLaserPoint` | `ourgoal/getLaserPoint` | 传入检测框像素,返回雷达拟合的目标物度量坐标 dx/dy/line_a |
| actionlib | `move_base` | `MoveBaseAction` | 所有点位导航(`sendPos` 封装) |

**通过参数服务器与其他节点握手**(均为全局参数,`init_params.launch` 启动时统一清零):

| 参数 | 写/读 | 对端 | 语义 |
|------|------|------|------|
| `awake2` | 读(阻塞) | `AI.py` | 语音唤醒+录音结束,放行底盘走迷宫 |
| `start_qr_scan` | 写 1 | `AI.py` | 到达首个扫码观察点,通知 AI 开始最终扫码 |
| `qr_scan_done` | 读 | `AI.py` | 三个二维码已找齐,停止旋转搜索 |
| `task1_all_done` | 读(阻塞) | `AI.py` | LLM 匹配与播报完成,放行后续阶段 |
| `real_item/real_class/real_room`、`sim_*` | 读 | `AI.py` | 任务一分配结果(物品/类别/车间) |
| `start_gazebo_sim` / `gazebo_sim_done` | 写/读 | 远程仿真端 | 仿真开始/完成 |
| `start_traffic_light_det` | 写 1 | `managed_nodes_client.py` | 联动启用 `traffic_light_ros` 与巡线节点(经各自 `set_enabled` 服务) |
| `vision_line_done` | 读(阻塞) | `find_way_ros` | 巡线 STOP 到位 |
| `start_vision_line` | 写 0 | 巡线管理 | 巡线结束后复位 |
| `CarX/CarY/CarYaw` | 读 | `TransformListener2` | TF 提取的地图位姿(缓存坐标换算用) |
| `auto_park_target/start_auto_park/auto_park_status` | 写 | 外部监控 | 仓储停车阶段的状态可见性 |

`/signal_class`、`/signal_detection` 的回调在 `target_locked_` 为 true 时**直接丢弃消息**——锁定后冻结识别结果,防止停车后新帧覆盖已被采用的目标数据。

### 各阶段详细流程

#### GotoA:迷宫逃脱(超声波 P 控制平移)

20 Hz 内部循环,全程 `angular.z = Kp_yaw × (0 − yaw)` 锁死航向。11 个子状态顺序执行,每个用超声波误差 P 控制(`Kp_dist=1.0`,限幅 `max_vel=0.5`),误差 < 0.05 m 切下一段:

```text
1 前(到 safe_F=0.15) → 2 右(0.20) → 3 前 → 4 左(0.20) → 5 前
→ 6 右(0.20) → 7 后(safe_B=0.35) → 8 左 → 9 后 → 10 右(safe_R2=0.25)
→ 11 解除锁头,固定 angular.z=1.57 原地转 45 个循环周期完成调头
```

结束发布零速刹车。此阶段与 AI.py 并行(AI 在做 LLM 匹配)。

#### GotoB:二维码扫码

1. 依次导航到 4 个硬编码观察点 `(-1.56,-0.50,3.14) / (-1.56,-0.70,-1.57) / (-1.56,-0.30,1.57) / (-1.56,-0.50,0)`,单点超时 15 s,超时/失败取消目标换下一点。
2. 到达首个成功点后停车 0.3 s,置 `start_qr_scan=1`(通知 AI.py 开始最终扫码)。
3. 每点最多 7 个视角(view 0 原地不转先扫,之后每轮原地转 60°:角速度 1.0 rad/s 约 1.05 s):静止 0.4 s 稳定 → 扫 1 s;期间轮询 `qr_scan_done==1` 即提前收工。
4. 兜底一:4 个点全没到 → 仍强制 `start_qr_scan=1`,避免 AI 永等。
5. 兜底二:所有点扫完仍不齐 → 原地 stop-and-look 无限循环(转 60° + 静止扫 1 s)直到 `qr_scan_done==1`。
6. 阻塞等 `task1_all_done==1` → 切 `XingHuoAI_`。

#### XingHuoAI:读取任务参数

读出并打印 `real_item/real_class/real_room` 与 `sim_item/sim_class/sim_room` 六个参数,无其他动作,切 `GOTOC1_`。

#### GotoC(target_num):生产区目标搜索与停车

`target_num=1` 为 GOTOC1(实体车,`real_class`),`2` 为 GOTOC2(仿真车,`sim_class`)。这是最复杂的阶段:

**① 目标解析**:车间名文本匹配出 `target_class`(含"食品"→0,"日用"→1,"电子/电/生产"→2);写 `auto_park_target` / `auto_park_status=IDLE`。

**② 生产区几何**:左上内墙角 `(-2.0, -1.3)`,格宽 0.5 m,10 列 × 4 行(全部可参数化 `production_*`,现场平移用)。

**③ 停车缓存优先**:`warehouse_parking_cache_[3]` 记住本次运行每个类别已验证过的停车点+观察点。命中缓存 → 直接 `navigateWithRetry` 到缓存停车点(15 s × 3 次),到达后跑一遍横向对正**现场复核**(对正成功=缓存仍有效,直接进入前后微调与播报;复核失败或不可达 → 作废缓存,落入观察点搜索)。

**④ 观察点生成**(两组,先粗后细):

- **粗搜(顺时针绕场 9 个位置)**:4 个共享角点(每个覆盖相邻两面墙各 2 格,先朝主墙识别、再原地转向次墙识别)+ 上墙中段每 2 格一个(3 个)+ 下墙中段每 3 格(2 个)。观察距离 `primary_view_distance=1.00 m`。
- **兜底(逐格 24 个位置)**:每个边界格单独一个近距离(`fallback_view_distance=0.50 m`)观察位。
- 每个位置派生 3 个候选:名义点 ± `lateral_retry_offset`(0.15 m,沿墙切向)。

**⑤ 候选筛选**(`selectObservationCandidate`):先做纯几何可见性(距离 ≤1.25 m、在 ±60° FOV 内),再用 `/move_base/global_costmap/costmap` 打视线遮挡(仅 occupancy≥100 的致命格算遮挡),再用 `/move_base/make_plan` 预规划判可达并取路径长度;按"无遮挡可见格多 → 路径短"选优。名义点已覆盖全区时不再规划其余候选。

**⑥ 到点与就地识别**:

- 导航 20 s 超时;**无论成败都先停车**、settle 0.4 s,并**清空行驶/转向期间的全部识别结果**(只认停车后的新帧)。
- 导航失败但当前位置几何上已能稳定看到本区域格(`currentPoseCanObserve`)→ 就地识别,否则换下一区域。
- 静止识别 2 s:要求 class 与 detection 消息均新鲜(<2 s)且 `current_signal_class_ == target_class`。**首次匹配即锁定**(`target_locked_=true`),锁定 center/left/right 像素。
- 角点第二面墙:清结果 → 原地转向(6 s 超时)→ settle → 重新识别。

**⑦ 度量定位与停车点计算**(锁定成功后):

1. 调 `/srv_getLaserPoint`(传中心/左/右像素)→ 雷达拟合出目标牌在**车体系**的 dx/dy 与 line_a,写 `signal_target_*` 参数。
2. 用 `CarX/CarY/CarYaw` 把目标旋转+平移到**地图系**。
3. 停泊墙判定:目标点到四面墙距离最小者;停车朝向**固定为正对该墙**(避免角落斜视/拟合误差造成斜停)。
4. 停车点 = 目标沿墙法向后退 0.30 m,再限幅到生产区内缩 0.30 m 的矩形内(限幅只挪点,不改朝向)。
5. 保存当前观察点 `last_parking_observation_*`(供 GotoD 失败恢复)。

**⑧ 导航停车 + 重试环**:`navigateWithRetry` 3 次 × 15 s;连续失败 → 无限重试"返回观察点(20 s)→ 再试停车点",直到成功(仅 `ros::ok` 能退出)。成功后写入停车缓存。

**⑨ 两段微调**(顺序执行,均失败也不阻断后续):

- **横向对正** `adjustLateralPosition`:只认停车后 <1 s 新鲜的 RKNN 检测;`linear.y = kp × (target_center_x − center_x)`(目标 320 px、容差 15 px、kp 0.0005、限 0.05 m/s),同时 yaw 锁定目标朝向(限 0.20 rad/s);连续 **5 个新检测帧**在容差内完成,超时 5 s。图像 x 增大方向与 linear.y 反号(图像右 = 车体左为正)。
- **前后对距** `adjustFrontDistance`:前方超声波调到 0.20 m(PID,限 0.15 m/s);每轮先等 `/ultra` 新样本(≤0.5 s),距离有效域 [0.05, 0.80] m,容差 0.02 m,连续 **3 个稳定样本**完成,超时 5 s。

微调结果写 `auto_park_status=DONE/FAILED`;**导航已成功,微调失败也放行播报**。

**⑩ 播报与转移**:GOTOC1 且找到目标 → `espeak "已将{real_item}放入{real_room}"`(播报2);搜遍所有区域未找到 → `auto_park_status=FAILED` 直接跳过。转移:GOTOC1→GOTOC2,GOTOC2→Gazebo_。

#### Gazebo:仿真协同

置 `gazebo_sim_done=0`、`start_gazebo_sim=1` → 10 Hz 轮询等远程 PC 置 `gazebo_sim_done=1` → 复位 `start_gazebo_sim=0` → `espeak "仿真任务已完成,已将{sim_item}放入{sim_room}"`(播报3)→ 切 `GOTOD_`。

#### GotoD:抵达巡线起点

1. 先复位 `start_traffic_light_det=0`(暂停红绿灯识别,防误触发)。
2. 若上一周期连观察点都没能返回(`goto_d_recovery_pending_`)→ 先回上次 GotoC 观察点,失败则本周期放弃,下周期再试。
3. 导航到**固定粗起点 (0.4, -3.08, yaw=-1.57)**,20 s × `gotod_nav_max_attempts`(3)次重试。
4. 失败 → 回上次 GotoC 观察点恢复,下个状态机周期重试粗起点。
5. 成功 → 清恢复标志,置 `start_traffic_light_det=1`:`managed_nodes_client.py` 轮询到后,联动启用 `traffic_light_ros`(识别一次方向后自闭)与巡线节点(参考上文 find_way_ros)→ 切 `VISION_LINE_`。

当前版本**跳过缺口对齐**(日志 "skip gap alignment"),靠固定粗起点。

#### vision_line:收尾

10 Hz 轮询等 `vision_line_done==1`(find_way_ros 跨停止线进 STOP 时置位)→ 置 `start_vision_line=0` → 停稳缓冲 2 s(满足"停后 10 秒内播报"规则)→ `espeak "任务完成"`(播报4)→ `ros::shutdown()`。

### 参数

**全局参数**(经公共 NodeHandle 读取,无 `~` 前缀;`init_params.launch` 只清握手标志,以下默认值均为代码内建):

| 参数 | 默认 | 说明 |
|------|------|------|
| `production_left_x` / `production_top_y` | -2.0 / -1.3 | 生产区左上内墙角(地图系) |
| `production_cell_size` / `production_columns` / `production_rows` | 0.5 / 10 / 4 | 格宽与网格规模 |
| `warehouse_primary_view_distance` / `warehouse_fallback_view_distance` | 1.00 / 0.50 | 粗搜/兜底观察距离 m |
| `warehouse_lateral_retry_offset` | 0.15 | 观察候选切向偏移 m |
| `warehouse_corner_view_inset` | 0.75 | 角点距墙内缩 m |
| `warehouse_observation_navigation_timeout` | 20.0 | 观察点导航超时 s |
| `warehouse_corner_turn_timeout` | 6.0 | 角点原地转向超时 s |
| `warehouse_stable_recognition_distance` | 1.25 | 稳定识别最大距离 m |
| `warehouse_stable_fov_degrees` | 120.0 | 稳定识别全 FOV 度 |
| `warehouse_observation_settle_duration` / `warehouse_observation_recognition_duration` | 0.4 / 2.0 | 停车稳定 / 静止识别时长 s |
| `warehouse_recognition_message_max_age` | 2.0 | 识别消息新鲜度 s |
| `warehouse_target_center_x` / `warehouse_center_tolerance_px` | 320 / 15 | 横向对正目标与容差 px |
| `warehouse_lateral_kp` / `warehouse_max_lateral_speed` / `warehouse_lateral_timeout` | 0.0005 / 0.05 / 5.0 | 横向对正增益/限速/超时 |
| `warehouse_front_kp`(ki/kd) | 1.0(0/0) | 前距 PID 增益 |
| `gotod_nav_max_attempts` / `gotod_nav_retry_delay` | 3 / 1.0 | GotoD 导航重试 |
| `gap_*` / `lidar_offset_*` | 见 core 构造 | 雷达缺口检测(当前未启用) |

**代码内建的固定值**(改需重新编译):迷宫安全距离 `safe_F/B/L/R/R2 = 0.15/0.35/0.20/0.20/0.25 m`、`Kp_dist=1.0`、`Kp_yaw=2.0`、`max_vel=0.5`;GotoB 四个观察点与 15 s 超时;GotoC 停车点后退 0.30 m、前后对距 0.20 m、横向稳定 5 帧、前距稳定 3 帧;GotoD 粗起点 (0.4, -3.08, -1.57) 与 20 s 超时。

### 保留未启用的部分

- `detectGap` + `ScanCallback` 采样(`gap_*` 参数族):为 GotoD 雷达缺口对齐准备,`collect_gap_samples_` 永远为 false,当前流程不触发。
- `class_command` 发布者、`/param_reload`、`/srv_getPosition`、`InitPID`、`calculateYVelocity` 等旧工具:未在当前状态机里调用。
- 头文件声明的 `GotoF` / `END` 无实现;`TEST_` / `END_` 状态无入口。

### 启动说明

- 可执行名 `switch_test2`(节点名 `switch_node`),随巡线/导航栈一起由 `startup_scripts` 的 launch 启动。
- 启动后节点会**阻塞在构造函数**直到 `awake2=1`,即 AI.py 完成唤醒录音后才会开始跑迷宫;调试时可直接 `rosparam set /awake2 1` 放行。
- 全流程靠参数握手推进,单步调试时注意 `init_params.launch` 的清零语义:roscore 不重启时旧标志会残留。

---


