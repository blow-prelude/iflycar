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

局域网内通过 SSH 或远程桌面连接板卡后，开机最初几分钟交互卡顿，发送命令
后不能及时响应。使用以下命令检查相关内核日志：

```bash
sudo dmesg |
grep -iE 'rtl|wlan|mmc1|sdio|timeout|crc|deauth|disconnect'
```

`wlan0` 最终可以正常进入可用状态：

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

使用 `/sbin/ethtool -i wlan0`、`modinfo RTL8821CS` 和 `/proc/modules`
检查得到：

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

#### 排查一：驱动内部省电

接口层查询 `iw dev wlan0 get power_save` 的结果为：

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

向 `/etc/modprobe.d/rtl8821cs.conf` 写入模块参数：

```bash
printf '%s\n' \
  'options RTL8821CS rtw_power_mgnt=0 rtw_ips_mode=0' |
sudo tee /etc/modprobe.d/rtl8821cs.conf
sudo reboot
```

重启后确认配置已经生效：

```text
rtw_power_mgnt = 0
rtw_ips_mode   = 0
```

`rtw_smart_ps` 和 `rtw_lps_level` 仍保留默认值是正常现象；它们只定义 LPS
启用时采用的方式，主开关 `rtw_power_mgnt=0` 后不会实际进入 LPS。

关闭驱动内部省电后，开机最初几分钟仍有卡顿，因此省电设置不是本次问题的
唯一根因。

#### 排查二：AP 和 STA 共用单射频

板卡同时运行了两个无线接口：

```text
phy#0
├─ wlan0  managed  wtrrr_5G  channel 60 / 5300 MHz / 40 MHz
└─ p2p0   AP       ucar-*    channel 60 / 5300 MHz / 40 MHz
```

但 `/etc/hostapd/hostapd.conf` 要求 p2p0 使用 2.4 GHz 信道 6：

```text
interface=p2p0
hw_mode=g
channel=6
country_code=CN
```

RTL8821CS 只有一个 `phy#0`。开机时 hostapd 先让 p2p0 使用 2.4 GHz 信道
6，随后 NetworkManager 让 wlan0 连接到 5 GHz 信道 60。驱动尝试将两个接口
同步到相同信道，最终把 p2p0 强制迁移到信道 60，并在连接过程中触发：

```text
WARNING: ... rtw_chset_sync_chbw+0xd4/0x144 [RTL8821CS]
rtw_join_done_chk_ch+0x15c/0x424 [RTL8821CS]
```

当前 SSH 实际使用 wlan0 的 `192.168.10.246`，没有通过 p2p0/br0 提供的
`10.42.0.1` 热点连接。停用不需要的热点并重启：

```bash
sudo systemctl disable --now hostapd
sudo reboot
```

重启后确认 hostapd 已停用、p2p0 已关闭：

```bash
systemctl is-enabled hostapd
systemctl is-active hostapd
ip -br address
iw dev
```

实际结果为 hostapd `disabled/inactive`、p2p0 `DOWN`，且内核日志中不再出现
`rtw_chset_sync_chbw()` WARNING。这一对照验证了该 WARNING 来自 RTL8821CS
的 AP+STA 跨频段并发。

如果以后需要恢复板卡热点：

```bash
sudo systemctl enable --now hostapd
```

需要同时保留热点和外部 Wi-Fi 时，应让 AP 与 STA 使用相同频段和信道，或者
增加第二块无线网卡分别承担 AP 和 STA，避免单射频跨信道并发。

#### 排查三：开机后的系统负载

关闭 AP 后，无线链路状态良好：

```text
signal: -38 dBm
rx bitrate: 135.0 MBit/s MCS 7 40MHz
tx bitrate: 150.0 MBit/s MCS 7 40MHz short GI
```

但开机约 4 分钟时系统仍出现高负载：

```text
load average: 11.53, 10.58, 4.71
iowait: 25%
```

进程检查显示主要资源占用来自：

```text
cpptools                         62.6%
NoMachine nxnode.bin            34.3%
Pylance                         19.0%
VS Code Remote extensionHost    16.9%
Xorg                            15.4%
NoMachine nxcodec.bin            9.4%
ToDesk                           6.6%
```

VS Code Remote 连接后，C/C++ 扩展、Pylance 和文件监视器会扫描整个工作区，
同时 NoMachine、Xorg 和 ToDesk 也在工作。CPU 使用和大量文件读取造成了开机
最初几分钟的高 load average 与 I/O wait，时间上与 SSH 卡顿一致。

可用以下命令复查：

```bash
uptime
vmstat 1 10
ps -eo pid,ppid,user,stat,etimes,comm,%cpu,%mem,args \
  --sort=-%cpu | head -30
```

关闭 VS Code Remote 窗口并断开不需要的 NoMachine、ToDesk 会话，只保留普通
终端 SSH 进行对照后，交互恢复正常。长期使用时应减少同时运行的远程桌面
服务，并限制 VS Code 对 `build`、`devel`、`install`、`log` 和 `.venv` 等
大型生成目录的索引与文件监视。

`systemd-analyze blame` 显示 `rc-local.service` 约耗时 16 秒，主要来自脚本中的
固定等待和 USB Hub 重启。`netfilter-persistent.service` 与
`ros_board_package.service` 虽然启动失败，但已经退出，不是持续占用 CPU 的
进程，因此不是本次开机后数分钟卡顿的主要原因。

#### 最终结论

本次问题包含两个独立因素：

1. p2p0 AP 与 wlan0 STA 共用 RTL8821CS 单射频并跨频段启动，导致驱动信道
   同步 WARNING。关闭不需要的 hostapd 后，该 WARNING 消失。
2. 关闭 AP 和驱动省电后，剩余的开机短时卡顿来自 VS Code Remote 索引及多个
   远程桌面服务叠加产生的 CPU、存储负载。减少这些后台任务后 SSH 恢复正常。

`host_wake_irq=0`、缺少 `ref_wifi_clk`/`sdio-supply`、USB `error -71` 以及
GPU/SquashFS 日志均没有在本次排查中表现出与 SSH 卡顿的直接关联。

---
### 离线中文语音播报方案

#### 需求与现象

系统自带的 espeak 可以离线播报中文，但合成声音电子感明显，部分物品名和车间名
吐字不够清楚。项目的播报内容又包含大模型动态生成的物品名称，不能全部替换成
提前录制的固定 WAV，因此需要一套完全离线、支持动态中文文本的 TTS。

目标运行环境为 ARM64、Debian 10，系统 CMake 版本为 `3.13`。语音方案不能要求
升级系统 glibc、libstdc++ 或整套编译工具链，也不能依赖比赛现场的互联网连接。

#### 尝试一：sherpa-onnx + Kokoro

sherpa-onnx 可以完全离线运行，支持多种 TTS 模型；Kokoro 支持中英文，并提供
INT8 量化模型，从功能和音质上能够满足需求。

实际接入时发现，能够支持中文 TTS 的相关发布版本至少要求 CMake `3.14.5`，而
本机只有 `3.13`。如果继续使用该方案，需要升级本机 CMake，或者额外维护一套
独立构建工具链，会扩大目标机环境改动范围。

本项目决定不为语音播报升级系统工具链，因此放弃该方案。

#### 最终方案：Piper

Piper 体积和运行开销较小，提供 ARM64 预编译版本，可以直接加载 ONNX 声音模型，
不需要在目标机重新编译推理框架。项目使用中文
`zh_CN-huayan-medium.onnx` 模型，配置中的采样率为 22050 Hz、质量级别为
`medium`。

当前完整运行时放在：

```text
3rdparty/piper/
├── piper
├── libonnxruntime.so.1.14.1
├── libespeak-ng.so -> libespeak-ng.so.1.1.51
├── libespeak-ng.so.1 -> libespeak-ng.so.1.1.51
├── libespeak-ng.so.1.1.51
├── espeak-ng-data/
└── models/
    ├── zh_CN-huayan-medium.onnx
    └── zh_CN-huayan-medium.onnx.json
```

`piper` 的 RUNPATH 包含 `$ORIGIN`，因此会优先从可执行文件所在目录加载
`libonnxruntime` 和 `libespeak-ng`，不需要把这些库安装到系统目录，也不需要修改
全局 `LD_LIBRARY_PATH`。

#### 版本兼容问题

Piper 预编译包并非都能运行在 Debian 10。较新的 ARM64 发布包在本机启动时曾出现：

```text
version `GLIBC_2.29' not found
version `GLIBCXX_3.4.26' not found
```

这表示该二进制是在更新的系统工具链上构建的，不是模型文件损坏。不要通过替换
系统 glibc 或 libstdc++ 强行运行，否则可能影响 ROS 和其他系统程序。

最终采用较老的 `v0.0.2` 预编译版本。该版本可适配当前系统，并使用
ONNX Runtime 1.14.1。迁移时必须同时保留 Piper 可执行文件、两个动态库、
`espeak-ng-data`、模型和模型配置，只复制 `piper` 与 `.onnx` 文件并不完整。

#### 基础验证

先确认动态库和程序本身能够启动：

```bash
ldd 3rdparty/piper/piper | grep 'not found'
3rdparty/piper/piper --help
```

第一条命令正常情况下没有输出。若出现 `not found`，应先补齐同目录动态库及符号
链接，不要继续测试模型。

使用实际中文模型进行合成：

```bash
cd 3rdparty/piper
printf '%s\n' '任务完成' |
./piper \
  --model models/zh_CN-huayan-medium.onnx \
  --config models/zh_CN-huayan-medium.onnx.json \
  --output_file /tmp/piper_test.wav
file /tmp/piper_test.wav
aplay --quiet /tmp/piper_test.wav
```

正常输出应为 16-bit、单声道、22050 Hz WAV。当前目标机实测模型加载约 0.5 秒，
短句推理约 0.5 秒；较长的物品分配播报推理约 1～2 秒，可以满足任务流程的播报
时机要求。

#### 项目中的使用情况

以下流程均优先使用 Piper：

- `switch_test2` 和 `switch_test2_pro`：实体仓储完成、仿真完成和最终任务完成播报；
- `AI.py`：ROS 流程中识别完二维码后的物品分配播报；
- `AI_no_ros.py`：独立 AI 流程中的物品分配播报。

Piper 合成失败、输出文件异常或 `aplay` 播放失败时，会自动使用 espeak 播报相同
文本。两种后端都失败时只记录错误，任务状态机继续执行，避免语音设备故障导致
整车流程永久阻塞。


#### 最终结论

在不升级 Debian 10 工具链的前提下，旧版 Piper 配合
`zh_CN-huayan-medium.onnx` 可以稳定提供离线中文动态播报。运行时依赖全部随项目
保存在 `3rdparty/piper/`，正常路径使用 Piper 改善清晰度，异常路径保留 espeak
兜底，兼顾播报效果和比赛流程可靠性。
