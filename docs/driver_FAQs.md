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

- 麦克风阵列内部已经设置了唤醒引擎。
- `/dev/ttyS3` 接收外部 AIUI 唤醒包。
- `uart_rec()` 完成串口分包和校验。
- `process_recv()` 解析 `eventType == 4` 的唤醒事件、获取角度，并将 ROS 参数 `awake` 设置为 `1`。
- 不调用 `hid_open()`、`recorder_creat()` 或 `set_awake_word()`。
- 当前语音节点不初始化 USB HID，也不负责 ALSA 录音；唤醒后的录音由监听 `awake=1` 的 Python 程序接管。


启动方式，可覆盖串口和波特率：

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
### 无线连接卡顿

#### 现象

局域网内通过 SSH 或远程桌面连接板卡后，交互过程中发送命令会出现卡顿。
`sudo dmesg | grep -i -E "wlan|wifi|firmware|under-voltage|error"`
系统启动日志中 `wlan0` 最终成功进入可用状态：

```text
[WLAN_RFKILL]: wlan_platdata_parse_dt: wifi_chip_type = rtl8821cs
IPv6: ADDRCONF(NETDEV_CHANGE): wlan0: link becomes ready
```

日志中没有发现欠压、Wi-Fi 固件加载失败或网卡断开记录。以下启动提示存在，
但目前没有证据表明它们直接导致 SSH 卡顿：

```text
[WLAN_RFKILL]: can't find rockchip,grf property
[WLAN_RFKILL]: WIFI,host_wake_irq = 0, flags = 0
[WLAN_RFKILL]: The ref_wifi_clk not found !
```

同一份日志中的 SquashFS 和 Mali GPU 报错与无线网络无关。`usb 2-1` 的
`error -71` 也不是该网卡的错误，因为无线网卡实际使用 SDIO 总线。

#### 已确认的网卡与驱动信息
输入 `/sbin/ethtool -i wlan0` 查看驱动信息 ，得到

```text
driver: rtl8821cs
version: v5.14.2-28-g6011b0372.20220328
firmware-version: 24.5
bus-info: mmc1:0001:1
kernel module: RTL8821CS
module file: /lib/modules/5.10.176/extra/RTL8821CS.ko
```

`RTL8821CS` 出现在 `/proc/modules` 中，并带有 `(O)` 标记，说明当前使用的是
可加载的树外厂商驱动，而不是内核内建驱动。`/lib/modules` 和
`/usr/lib/modules` 下显示的模块路径通常来自 usr-merge，是同一份模块，不代表
加载了两个相互冲突的驱动。

接口层查询 `iw dev wlan0 get power_save` ，结果为：

```text
Power save: off
```

但驱动模块参数为：

```text
rtw_power_mgnt    = 2
rtw_ips_mode      = 1
rtw_smart_ps      = 2
rtw_lps_level     = 1
rtw_low_power     = 0
rtw_lps_chk_by_tp = 1
rtw_en_napi       = 1
rtw_en_gro        = 1
```

其中 `rtw_power_mgnt=2` 表示驱动内部采用较积极的省电模式，
`rtw_ips_mode=1` 表示 IPS 已启用，`rtw_lps_level=1` 对应 SDIO 低时钟省电
状态。NAPI 和 GRO 已启用，通常不是交互卡顿的原因。

#### 解决方案
向配置文件 `/etc/modprobe.d/8821cs.conf` 里写入非省电配置

```test
options RTL8821CS rtw_power_mgnt=0 rtw_ips_mode=0
```

关闭积极的省电模式

#### 当前分析结论

网卡已成功初始化和关联，现有日志不足以证明设备树提示、固件或 USB 错误是
SSH 卡顿的直接原因。接口层报告省电关闭，但 Realtek 厂商驱动内部仍配置了
LPS/IPS，因此驱动从空闲省电状态恢复时产生延迟是当前的主要怀疑方向，尤其
符合“空闲后第一条命令卡顿、随后短时间恢复正常”的表现。

该判断目前仍是排查结论，不是已经验证的根因。需要通过修改驱动加载参数前后
的对照测试，并结合网关 ping 延迟、`iw dev wlan0 station dump` 中的重传和
失败计数，才能区分省电唤醒、无线信号干扰和 SDIO 传输问题。目前尚未形成
经过验证的最终解决方案。