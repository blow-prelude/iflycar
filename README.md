<h2 align="center" style="color: skyblue;">2026年智能车创意赛道讯飞组 ——</h2>
<h2 align="center" style="color: skyblue;">智慧工厂</h2>

### 环境
- 上位机：rk3588s，arm64位
- 操作系统：debian10
- ROS noetic
- python：ROS预装3.8， 部分节点使用虚拟环境3.9

---
### 功能包功能简介
#### camera_2d_lidar_calibration
RGB相机和二维雷达联合标定

#### car_sr
已弃用。原本用于arm开发板和局域网内其他设备的TCP通信

#### costmap_prohibition_layer
放置虚拟墙、禁止区

#### fdilink_ahrs
imu驱动

#### find_signal
用ocr_rknn推理，识别文字

#### geometry
用于tf

#### geometry2
用于tf2

#### navigation
用于导航+路径规划的第三方库

#### object_information_msgs
ocr推理收发有效数据的自定义消息包

#### ourgoal
前往目标点的业务逻辑

#### pcl_work
激光雷达数据转pcl点云数据

#### rknn
已弃用。原本是上一年的rknn推理代码

#### rknn_ros
已弃用。原本是上一年的rknn推理的ros部署

#### speech_command
麦克风阵列驱动，串口唤醒

#### startup_scripts
提供一键启动、启动参数初始化和视觉节点运行状态管理

#### traffic_light
用rknn推理+cv过滤，识别交通灯

#### ucar_camera
RGB相机节点

#### ucar_controller
无刷电机、LED、IMU的控制包

#### ucar_map
SLAM建图

#### ucar_nav
导航

#### vision_line
视觉巡线+控制命令

#### vision_ros
已弃用。原本是上一年的视觉巡线的ros端部署

#### ydlidar
激光雷达驱动

---
### quick start
一键启动 `./start_all.sh`
启动巡线测试 `roslaunch startup_scripts start_vision_line.launch`

---
### `managed_nodes_client.py` 节点生命周期管理

`startup_scripts/scripts/managed_nodes_client.py` 通过各节点提供的
`std_srvs/SetBool` 服务统一切换视觉节点的运行状态。这里的“生命周期管理”
指启用或禁用节点的业务处理；被禁用的 ROS 进程仍然存活，服务仍可调用，
不会被脚本启动、终止或重启。

`start_all.launch` 启动名为 `/managed_nodes` 的管理节点。默认管理关系如下：

| 逻辑名称 | 服务配置参数 | 默认 `SetBool` 服务 | 作用 |
| --- | --- | --- | --- |
| `traffic_light_ros` | `~traffic_light_service` | `/traffic_light_ros/set_enabled` | 交通灯检测 |
| `image_process` | `~image_process_service` | `/image_process/set_enabled` | 视觉巡线图像处理 |
| `vision_line_node` | `~vision_line_service` | `/vision_line_node/set_enabled` | 视觉巡线控制 |
| `find_signal` | `~find_signal_service` | `/find_signal/set_enabled` | OCR 标志牌识别 |

服务名称可通过表中的管理节点私有参数覆盖。调用服务前最多等待
`~service_wait_timeout` 秒，默认值为 2 秒；调用失败时不更新状态缓存，后续
轮询会继续尝试。

#### 参数驱动的状态切换

管理节点默认以 10 Hz（`~parameter_poll_rate`）读取以下 ROS 参数：

- `~traffic_light_enable_param`：交通灯检测开关参数名，默认指向
  `/start_traffic_light_det`。
- `~vision_line_enable_param`：视觉巡线开关参数名，默认与交通灯开关参数相同。
- `~find_signal_enable_param`：OCR 请求参数名，默认指向 `/task1_all_done`。
- `~manage_find_signal`：是否管理 OCR 节点，默认值为 `true`。

`start_all.launch` 将交通灯和视觉巡线开关都绑定到
`/start_traffic_light_det`，因此该启动方式下它们仍然同步启停。

OCR 与交通灯检测互斥，`find_signal` 的实际启用条件为：

```text
find_signal_enabled = task1_all_done && !start_traffic_light_det
```

完整状态关系如下：

| `/start_traffic_light_det` | `/task1_all_done` | 交通灯和巡线节点 | `find_signal` |
| ---: | ---: | --- | --- |
| 0 | 0 | 禁用 | 禁用，静默等待 |
| 0 | 1 | 禁用 | 启用 |
| 1 | 0 | 启用 | 禁用，静默等待 |
| 1 | 1 | 启用 | 禁用，静默等待 |

当 `/start_traffic_light_det` 变为 `1` 时，管理节点会在启用
`traffic_light_ros` 前禁用 `find_signal`，避免 OCR 与交通灯推理同时占用计算
资源。当该参数恢复为 `0` 时，先禁用交通灯，再根据 `/task1_all_done` 决定
是否恢复 OCR。视觉巡线节点独立服从 `~vision_line_enable_param` 指向的参数。

`init_params.launch` 在整套系统启动时将两个参数都初始化为 `0`，因此所有
受管视觉处理默认处于禁用状态。`find_signal` 被禁用后仍接收图像回调，但会
直接返回，不进行 OCR 推理或发布识别结果，表现为静默等待。

可手动检查或切换参数：

```bash
rosparam get /start_traffic_light_det
rosparam get /task1_all_done
rosparam set /start_traffic_light_det 1
rosparam set /task1_all_done 1
```

#### 手动管理服务

管理节点还提供 `/managed_nodes/set_enabled` 服务。该服务仅统一控制
`traffic_light_ros`、`image_process` 和 `vision_line_node`，不控制
`find_signal`，也不会修改上述两个 ROS 参数。启用时按照表中顺序调用三个
节点；禁用时按相反顺序调用。

```bash
rosservice call /managed_nodes/set_enabled "data: true"
rosservice call /managed_nodes/set_enabled "data: false"
```

参数轮询和手动服务调用使用同一把锁进行串行化，避免多个启停请求同时修改
受管节点状态。

#### 单独启动巡线任务

`start_vision_line.launch` 将视觉巡线开关单独绑定到始终为 `true` 的
`/start_vision_line_enabled`，所以启动后会立即启用 `image_process` 和
`vision_line_node`。交通灯 ROS 进程始终启动，但检测业务由
`/start_traffic_light_det` 决定，默认启用：

```bash
# 启动巡线，默认同时启用交通灯检测
roslaunch startup_scripts start_vision_line.launch

# 启动巡线，但不启用交通灯检测业务
roslaunch startup_scripts start_vision_line.launch traffic_light_enabled:=false
```

运行期间也可以独立切换交通灯业务，不会关闭巡线：

```bash
rosparam set /start_traffic_light_det 0
rosparam set /start_traffic_light_det 1
```

该 launch 没有启动 `find_signal`，因此给管理节点设置了
`~manage_find_signal=false`，不会等待或调用不存在的 OCR 服务。

---
###　外设问题

[麦克风阵列不能识别唤醒词](./docs/driver_FAQs.md#speech_command-找不到麦克风设备问题)
[网络质量太差，ssh连不上](./docs/driver_FAQs.md#无线连接卡顿)

---
### 更新日志
[视觉巡线更新日志](./src/vision_line/README.md)
