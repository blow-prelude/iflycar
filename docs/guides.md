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




