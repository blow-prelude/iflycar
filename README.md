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

管理节点默认以 10 Hz（`~parameter_poll_rate`）读取以下两个 ROS 参数：

- `/start_traffic_light_det`：控制交通灯检测和两个视觉巡线节点。
- `/task1_all_done`：请求启用 `find_signal` OCR。

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

当 `/start_traffic_light_det` 变为 `1` 时，管理节点先禁用 `find_signal`，
再依次启用 `traffic_light_ros`、`image_process` 和 `vision_line_node`，避免
OCR 与交通灯推理同时占用计算资源。当该参数恢复为 `0` 时，先禁用交通灯和
巡线节点，再根据 `/task1_all_done` 决定是否恢复 OCR。

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

---
### `speech_command` 找不到麦克风设备问题

#### 现象

运行语音节点时曾出现：

```text
>>>>>正在初始化麦克风阵列 HID
>>>>>找不到麦克风设备
>>>>>无法打开麦克风阵列 HID：未找到设备
>>>>>麦克风阵列 HID 未启动，硬件唤醒不可用
```

但使用下面的命令可以正常录音：

```bash
arecord -D hw:3,0 -f S16_LE -r 16000 -c 1 test.wav
```

#### 原因

日志中的“找不到麦克风设备”并不是 ALSA 录音设备打开失败，而是厂商的 USB HID 控制库没有找到兼容设备。这是两条互相独立的硬件通道：

- ALSA PCM 接口负责采集音频，对应 `hw:3,0`。
- USB HID/厂商控制接口负责设置唤醒词、获取阵列角度和控制灯光。

现场设备已经被 Linux 正常识别：

```text
Bus 002 Device 006: ID 2207:0001 Fuzhou Rockchip Electronics Company
card 3: XFMDPV0018 [XFM-DP-V0.0.18], device 0: USB Audio
```

其 USB 接口为：

```text
Interface 0: Vendor Specific Class
Interface 1: Audio, Driver=snd-usb-audio
Interface 2: Audio, Driver=snd-usb-audio
```

项目中的 ARM64 `libhid_lib.so` 固定查找 `VID:PID = 10d6:b003`，而实际设备为 `2207:0001`。厂商库会先比较 VID/PID，匹配后才调用 `libusb_open()`，因此本次失败发生在权限检查之前，不是 libusb 或 udev 权限问题。

没有 `/dev/hidraw*` 也不代表 USB 麦克风没有被识别。当前设备的控制接口属于 `Vendor Specific Class`，不是标准 HID Class；而且该厂商库直接通过 libusb 访问 `/dev/bus/usb`，不依赖 `hidraw` 节点。

可使用以下命令复查：

```bash
lsusb
lsusb -t
arecord -l
ls -l /dev/snd/
lsusb -d 10d6:b003
```

如果 `arecord` 正常、`lsusb` 中存在 `2207:0001`，但 `lsusb -d 10d6:b003` 没有输出，即可确认是设备型号或厂商库不匹配。修改 udev 权限规则不能解决 VID/PID 不匹配，不建议直接修改二进制库中的 VID/PID，以免错误占用 USB Audio 接口。

#### 最终方案：纯串口唤醒

项目决定参考提交 `0c63c6d94dfd07f0895b5b5d592e11e1c89a7de2`，采用纯串口唤醒：

- 麦克风阵列内部可能已经设置了唤醒引擎。
- `/dev/ttyS3` 接收外部 AIUI 唤醒包。
- `uart_rec()` 完成串口分包和校验。
- `process_recv()` 解析 `eventType == 4` 的唤醒事件、获取角度，并将 ROS 参数 `awake` 设置为 `1`。
- 不调用 `hid_open()`、`recorder_creat()` 或 `set_awake_word()`。
- 当前语音节点不初始化 USB HID，也不负责 ALSA 录音；唤醒后的录音由监听 `awake=1` 的 Python 程序接管。

启动方式：

```bash
source /home/ucar/ucar_ws/devel/setup.bash
roslaunch speech_command speech_command.launch
```

可覆盖串口和波特率：

```bash
roslaunch speech_command speech_command.launch \
  serial_port:=/dev/ttyS3 \
  baud_rate:=115200
```

正常启动时应看到：

```text
唤醒方式：纯串口（不初始化 USB HID 和 ALSA 录音）
纯串口唤醒模式启动，正在监听唤醒信号...
```


---
### 更新日志
[视觉巡线更新日志](./src/vision_line/README.md)
