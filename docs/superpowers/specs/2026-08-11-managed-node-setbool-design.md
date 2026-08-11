# 三个业务节点统一启停服务设计

## 目标

在 `startup_scripts` 包中新增一个节点，统一管理下列三个既有节点的业务运行状态：

- `/traffic_light`：`traffic_light/traditional_light_cv_ros.py`
- `/image_process`：`vision_line/find_way_ros`
- `/vision_line_node`：`vision_line/vision_line_node_fixed`

三个业务节点分别提供 `std_srvs/SetBool` 服务。管理节点持有三个服务客户端，并通过一个统一的 `std_srvs/SetBool` 入口批量启用或停用三个节点。

这里的“启停”只控制业务逻辑，不启动、关闭或重启 ROS 进程：节点停用后仍保持 ROS 通信和服务回调能力，以便随时恢复。

## 范围

本次后续实现包含：

- 在三个业务节点中增加 `SetBool` 服务端和运行状态。
- 在 `startup_scripts` 中增加统一管理节点。
- 将管理节点加入 `start_all.launch`。
- 补充三个包的 `std_srvs` 依赖以及脚本安装/构建配置。
- 验证重复调用、部分服务缺失、停用期间无业务输出以及恢复运行。

本次不包含：

- 使用 `roslaunch` API 杀死或重启进程。
- 管理清单之外的其他节点。
- 新增自定义 service 消息。
- 持久化上一次启停状态；进程重启后使用启动参数指定的初始值。

## 总体架构

```text
rosservice call /managed_nodes/set_enabled "data: true|false"
                         |
                         v
       startup_scripts/managed_nodes_client.py
          |              |                 |
          v              v                 v
/traffic_light/   /image_process/   /vision_line_node/
 set_enabled       set_enabled        set_enabled
     SetBool           SetBool            SetBool
```

统一管理节点同时承担两个 ROS 角色：

- 对外是 `/managed_nodes/set_enabled` 的 `SetBool` 服务端，提供单一管理入口。
- 对内是三个业务服务的 `SetBool` 客户端，依次转发同一个布尔目标状态。

这样既满足“在 `startup_scripts` 下创建服务客户端”的要求，也避免只能在管理节点源码中硬编码某个时刻调用、却没有外部控制入口的问题。

## ROS 接口

### 统一管理服务

```text
服务名：/managed_nodes/set_enabled
类型：  std_srvs/SetBool
请求：  data=true  启用三个节点
        data=false 停用三个节点
响应：  success=true  三个目标服务均调用成功并接受目标状态
        success=false 至少一个目标服务不可用、调用失败或拒绝请求
```

响应 `message` 汇总每个节点结果，例如：

```text
traffic_light=ok; image_process=ok; vision_line_node=unavailable
```

### 三个业务服务

| 节点 | 服务名 | 类型 |
|---|---|---|
| `/traffic_light` | `/traffic_light/set_enabled` | `std_srvs/SetBool` |
| `/image_process` | `/image_process/set_enabled` | `std_srvs/SetBool` |
| `/vision_line_node` | `/vision_line_node/set_enabled` | `std_srvs/SetBool` |

服务使用节点私有名 `~set_enabled` 注册，因而在当前 launch 节点名下解析为表中的全局名称。管理端服务名称做成私有参数，默认使用上述值，方便以后 remap 或替换节点名。

所有回调均满足幂等性：重复设置为当前状态也返回 `success=true`，`message` 说明节点已经处于目标状态。

## 公共状态语义

三个节点各自维护线程安全的 `enabled` 状态。交通灯和 `find_way_ros` 固定以 disabled 启动，并由管理节点根据同一个 `/start_traffic_light_det` 参数分别调用各自的 `SetBool` 服务；运动控制节点的初始值从私有参数 `~initially_enabled` 读取，默认 `true`。

服务回调只负责快速完成状态切换和必要的状态清理，不在回调中执行耗时业务逻辑。主循环每轮仍处理 ROS 回调，然后判断 `enabled`：

```text
ROS 回调处理
  -> enabled=false：执行一次停用收尾，按低频率休眠
  -> enabled=true：执行原有业务逻辑，按原频率休眠
```

“休眠等待”不能使用无限阻塞条件变量或停止 `spin/spinOnce`，否则单线程节点无法收到下一次 `SetBool(true)`。停用循环频率默认 10 Hz，可通过各节点私有参数 `~disabled_rate` 调整。

状态切换时只记录一条 INFO 日志，停用循环中不逐轮打印，避免刷屏。

## 各业务节点改动

### `traditional_light_cv_ros.py`

在 `TrafficLightRosNode` 中注册 `~set_enabled`，用锁保护 `_enabled`。图像订阅和服务始终存在；停用后图像回调不再做 `CvBridge` 转换或更新最新帧，主循环只休眠。

从启用切换到停用时：

- 清空最新帧，避免恢复后立即处理停用前的旧图像。
- 重置 FPS、检测稳定性/日志跟踪等跨帧状态。
- 若 OpenCV 窗口已创建则关闭窗口。

从停用切换到启用时，从新到达的相机帧开始检测。

`traditional_light_cv_ros.py` 不再直接读取 `start_traffic_light_det`。该参数由统一管理节点监听，并在值发生变化时调用交通灯的 `SetBool` 服务：

```text
start_traffic_light_det=1 -> /traffic_light/set_enabled(data=true)
                              /image_process/set_enabled(data=true)
start_traffic_light_det=0 -> /traffic_light/set_enabled(data=false)
                              /image_process/set_enabled(data=false)
```

交通灯业务节点只判断服务维护的 `enabled` 状态。管理节点不修改该参数，避免形成参数和服务之间的反馈循环。

当前节点在发布 `straight`、`left` 或 `right` 后会调用 `rospy.signal_shutdown`。为支持后续再次启用，改为完成发布等待后将自身业务状态切为 disabled，保持进程和服务存活。再次收到 `SetBool(true)` 后可进入下一轮检测。

### `find_way_ros.cpp`（节点名 `/image_process`）

在 `FindWayROS` 中注册 `~set_enabled`，使用互斥锁或原子变量保存 `enabled_`。循环开头保留 `ros::spinOnce()`；停用时跳过图像克隆、图像处理、状态机推进、话题发布和 STOP 延迟退出判断，只按停用频率休眠。

从启用切换到停用时：

- 清空 `latest_frame_`，防止恢复后处理旧帧。
- 将视觉状态机复位到 `IDLE`。
- 清除方向重置标志、停止线计数、上一条有效输出缓存以及 FPS 计数。
- 将中线模式恢复为 `MID_AVG`。
- 将 `/start_vision_line2` 设为 `0`，避免恢复时继承半途转弯状态。

恢复后等待新图像，并根据当前方向配置从干净状态重新运行。

### `vision_line_fixed.cpp`（节点名 `/vision_line_node`）

在 `VisionErrorController` 中注册 `~set_enabled`。主循环无论启停都执行 `ros::spinOnce()`；停用时不调用 `processVisionData()`，只休眠。

该节点会发布 `/cmd_vel`，因此停用必须具有安全收尾：

- 在 `true -> false` 的服务回调中立即发布一次全零 `geometry_msgs/Twist`，不能等到下一轮主循环。
- 清空尚未消费的视觉数据和 PID 积分/微分记忆。
- 将机动状态复位到 `IDLE`，清除转弯、稳定计数和当前误差，防止恢复后继续之前的 FORWARD/ROTATE 计时动作。
- 停用期间方向和视觉订阅回调不更新业务缓存，避免恢复时使用停用期间积累的指令或数据。

恢复后必须等待新的视觉数据或新的方向指令才产生非零控制量。

## 管理节点行为

新增：

```text
src/startup_scripts/scripts/managed_nodes_client.py
```

节点名为 `managed_nodes`。启动时创建三个持久或普通 `ServiceProxy`，并注册统一服务。服务调用采用有限等待，私有参数如下：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `~traffic_light_service` | `/traffic_light/set_enabled` | 交通灯服务名 |
| `~image_process_service` | `/image_process/set_enabled` | 图像处理服务名 |
| `~vision_line_service` | `/vision_line_node/set_enabled` | 运动控制服务名 |
| `~service_wait_timeout` | `2.0` | 每个服务最大等待秒数 |
| `~traffic_light_enable_param` | `/start_traffic_light_det` | 交通灯启停来源参数 |
| `~parameter_poll_rate` | `10.0` | 参数变化检查频率 |

批量调用顺序按安全性确定：

- 停用：`vision_line_node -> image_process -> traffic_light`。先让运动控制立即发布零速度。
- 启用：`traffic_light -> image_process -> vision_line_node`。先恢复数据生产端，最后恢复可能发布运动指令的控制端。

每个目标都独立尝试。某个服务失败后仍继续调用其余服务，以尽量达到安全目标；最终统一响应返回 `success=false` 并列出失败项。

不执行自动回滚。分布式服务调用不存在原子提交，回滚既可能再次失败，也可能在停用请求中重新启用运动节点，违反安全优先原则。调用方可在故障排除后重复发送同一目标状态，利用幂等性收敛。

管理节点启动时不主动改变三个节点的状态，避免启动先后顺序导致意外停机或启车。状态只在收到统一服务请求后改变。

## 启动与依赖

### `startup_scripts`

- `package.xml` 增加 `rospy` 和 `std_srvs` 运行依赖。
- `CMakeLists.txt` 使用 `catkin_install_python` 安装管理脚本。
- `start_all.launch` 在三个业务节点之后启动 `managed_nodes_client.py`。
- 交通灯和 `find_way_ros` 固定以 disabled 启动，并等待管理节点根据 `/start_traffic_light_det` 同步调用两个服务；运动控制节点显式设置 `~initially_enabled=true`。

### `traffic_light`

- `package.xml` 增加 `std_srvs` 运行依赖。
- Python 代码导入 `std_srvs.srv.SetBool` 和 `SetBoolResponse`。

### `vision_line`

- `package.xml` 增加 `std_srvs` 的 build、build_export 和 exec 依赖。
- `CMakeLists.txt` 的 `find_package(catkin REQUIRED COMPONENTS ...)` 增加 `std_srvs`，两个 C++ 目标继续链接 `${catkin_LIBRARIES}`。
- 两个 C++ 源文件包含 `std_srvs/SetBool.h`。

## 并发与回调约束

当前三个节点均使用单线程回调处理模型，因此 `enabled` 通常不会与业务循环并发修改；仍保留锁/原子状态以防未来切换到异步 spinner，也用于保护图像和控制状态的组合清理。

锁的使用遵循以下规则：

- 服务回调不在持锁时执行 `sleep`、等待其他服务或进行图像处理。
- 图像帧锁、方向锁和运行状态锁不嵌套；需要清理多个状态时使用固定顺序并缩短临界区。
- 管理节点串行调用三个服务，避免多个服务线程同时修改共享诊断结果。

## 故障处理

- 目标服务不存在或超时：继续处理其余节点，统一响应为失败并记录具体服务名。
- 服务传输异常：捕获 `rospy.ServiceException`，节点本身不退出。
- 业务节点收到重复状态：返回成功，不重复执行清理或重复发布零速度。
- 业务处理抛出异常：沿用各节点现有异常策略；管理服务不吞掉业务异常。
- `/vision_line_node/set_enabled false` 的零速度发布完成后才返回成功。
- 管理节点退出不会改变业务节点当前状态；它只负责发命令，不作为运行许可心跳。

## 测试与验收

### 静态与构建检查

1. Python 文件通过 `python3 -m py_compile`。
2. `catkin_make --pkg startup_scripts traffic_light vision_line` 构建成功。
3. `rosservice type` 显示四个服务均为 `std_srvs/SetBool`。

### 服务端测试

对每个业务服务分别验证：

1. `data=false` 返回成功，节点进程仍存在，服务仍可查询。
2. 重复 `data=false` 返回成功且不产生重复副作用。
3. 停用期间不执行图像检测/图像处理/控制业务，也不发布新的业务输出。
4. `/vision_line_node` 停用时立即发布一次零 `/cmd_vel`。
5. `data=true` 后只消费新到达的数据并恢复业务。
6. 重复 `data=true` 返回成功。

### 统一管理测试

```bash
rosservice call /managed_nodes/set_enabled "data: false"
rosservice call /managed_nodes/set_enabled "data: true"
```

验收结果：

- 两次调用在三个服务均在线时返回 `success: true`。
- 停用后三个进程和四个服务仍在线，运动控制已输出零速度。
- 启用后三个节点恢复业务处理。
- 人为关闭任一业务节点后，统一调用返回 `success: false`、消息指出故障节点，其余在线节点仍完成目标状态切换。

### 回归检查

- 交通灯和 `find_way_ros` 启动后保持 disabled，直到管理节点处理 `/start_traffic_light_det`；运动控制节点保持 `initially_enabled=true` 的既有启动行为。
- 交通灯发布方向后不再退出，服务仍在线，且业务自动进入 disabled 等待状态。
- 多次启停后不存在旧图像、旧方向、PID 积分或未完成机动动作被继续使用的现象。

## OCR 参数触发扩展

`startup_scripts/managed_nodes_client.py` 额外监听 `/task1_all_done`，并将其状态转发到 `/find_signal/set_enabled`。`find_signal/rknn_ros.cpp` 提供该 `std_srvs/SetBool` 服务，默认 disabled；只有 enabled 时图像回调才向 PPOCR 推理池提交图像。

`start_all.launch` 使用 C++ 可执行文件 `find_signal/rknn_ros`，不再启动同名功能的 Python 脚本。参数映射可通过管理节点私有参数 `~find_signal_enable_param` 和 `~find_signal_service` 覆盖。

## 实施文件清单

新增：

- `src/startup_scripts/scripts/managed_nodes_client.py`

修改：

- `src/startup_scripts/CMakeLists.txt`
- `src/startup_scripts/package.xml`
- `src/startup_scripts/launch/start_all.launch`
- `src/traffic_light/scripts/traditional_light_cv_ros.py`
- `src/traffic_light/package.xml`
- `src/vision_line/src/find_way_ros.cpp`
- `src/vision_line/src/vision_line_fixed.cpp`
- `src/vision_line/CMakeLists.txt`
- `src/vision_line/package.xml`
