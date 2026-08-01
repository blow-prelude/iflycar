# ucar_camera C++ ROS 节点 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `ucar_camera` 包里实现一个 C++ ROS 节点 `camera_ros`，对标 [scripts/ucar_camera.py](src/ucar_camera/scripts/ucar_camera.py)，调用 [CameraCapture](src/ucar_camera/include/ucar_camera.h) 完成 V4L2 采集 → 可选去畸变 → 水平翻转 → 发布 `sensor_msgs/Image`。

**Architecture:** 单一可执行 `camera_ros`，把 `ucar_camera.cpp` 直接编进可执行（不单独做库）。节点内部定义 `CameraRosNode` 类，放匿名命名空间；构造函数读 ROS 参数、构造 `CameraCapture`、建 publisher，`run()` 跑采集-处理-发布循环。手动填 `sensor_msgs/Image` 字段（不用 cv_bridge，与 Python 对齐）。

**Tech Stack:** ROS（roscpp / std_msgs / sensor_msgs）、OpenCV、catkin / colcon。无单测框架，靠 `colcon build` + `rosrun` + `rostopic` + `rqt_image_view` 验证。

---

## 文件结构

| 文件 | 类型 | 职责 |
|---|---|---|
| [src/ucar_camera/package.xml](src/ucar_camera/package.xml) | 修改 | 三联依赖加 `sensor_msgs`、`libopencv-dev` |
| [src/ucar_camera/CMakeLists.txt](src/ucar_camera/CMakeLists.txt) | 修改 | `find_package` 加 `sensor_msgs` / `OpenCV`；`include_directories` 加 `include`；新增 `camera_ros` target |
| [src/ucar_camera/src/camera_ros.cpp](src/ucar_camera/src/camera_ros.cpp) | 改写（原 1 行空文件） | `CameraRosNode` 类 + `main`，约 70 行 |

不动 [ucar_camera.h](src/ucar_camera/include/ucar_camera.h) / [ucar_camera.cpp](src/ucar_camera/src/ucar_camera.cpp) / [scripts/ucar_camera.py](src/ucar_camera/scripts/ucar_camera.py)。

---

## Task 1: 更新 package.xml 依赖

**Files:**
- Modify: [src/ucar_camera/package.xml](src/ucar_camera/package.xml)

- [ ] **Step 1: 在 `<build_depend>` 分组追加 sensor_msgs 和 libopencv-dev**

把 [package.xml:53-55](src/ucar_camera/package.xml#L53-L55) 的 build_depend 段从：

```xml
  <build_depend>roscpp</build_depend>
  <build_depend>rospy</build_depend>
  <build_depend>std_msgs</build_depend>
```

改为：

```xml
  <build_depend>roscpp</build_depend>
  <build_depend>rospy</build_depend>
  <build_depend>std_msgs</build_depend>
  <build_depend>sensor_msgs</build_depend>
  <build_depend>libopencv-dev</build_depend>
```

- [ ] **Step 2: 在 `<build_export_depend>` 分组追加同样两行**

把 [package.xml:57-59](src/ucar_camera/package.xml#L57-L59) 从：

```xml
  <build_export_depend>roscpp</build_export_depend>
  <build_export_depend>rospy</build_export_depend>
  <build_export_depend>std_msgs</build_export_depend>
```

改为：

```xml
  <build_export_depend>roscpp</build_export_depend>
  <build_export_depend>rospy</build_export_depend>
  <build_export_depend>std_msgs</build_export_depend>
  <build_export_depend>sensor_msgs</build_export_depend>
  <build_export_depend>libopencv-dev</build_export_depend>
```

- [ ] **Step 3: 在 `<exec_depend>` 分组追加同样两行**

把 [package.xml:61-63](src/ucar_camera/package.xml#L61-L63) 从：

```xml
  <exec_depend>roscpp</exec_depend>
  <exec_depend>rospy</exec_depend>
  <exec_depend>std_msgs</exec_depend>
```

改为：

```xml
  <exec_depend>roscpp</exec_depend>
  <exec_depend>rospy</exec_depend>
  <exec_depend>std_msgs</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>libopencv-dev</exec_depend>
```

- [ ] **Step 4: 校验 XML 格式正确**

Run: `xmllint --noout src/ucar_camera/package.xml`（或直接肉眼检查标签闭合）
Expected: 无输出 / 标签无错位

> 本任务的 package.xml 改动不单独提交，与下一任务 Task 2 的 CMakeLists.txt 改动一起在 Task 2 Step 5 提交（都属于 build 配置变更）。

---

## Task 2: 更新 CMakeLists.txt 构建

**Files:**
- Modify: [src/ucar_camera/CMakeLists.txt](src/ucar_camera/CMakeLists.txt)

- [ ] **Step 1: `find_package` 加 sensor_msgs**

把 [CMakeLists.txt:10-14](src/ucar_camera/CMakeLists.txt#L10-L14) 从：

```cmake
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)
```

改为：

```cmake
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
)
```

- [ ] **Step 2: 新增 OpenCV 查找**

在 Step 1 改完的 `find_package(catkin ...)` 块下方插入一行（紧跟第 14 行 `)` 之后）：

```cmake

find_package(OpenCV REQUIRED)
```

- [ ] **Step 3: `include_directories` 加 include 和 OpenCV**

把 [CMakeLists.txt:103-106](src/ucar_camera/CMakeLists.txt#L103-L106) 从：

```cmake
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
```

改为：

```cmake
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
```

注意：`include` 前面去掉注释符号 `#`。

- [ ] **Step 4: 新增 camera_ros 可执行 target**

在 [CMakeLists.txt:108-111](src/ucar_camera/CMakeLists.txt#L108-L111) 的 "Declare a C++ library" 注释段之后、"Add cmake target dependencies" 段之前，插入：

```cmake

## camera_ros 可执行节点
add_executable(camera_ros
  src/camera_ros.cpp
  src/ucar_camera.cpp
)
target_link_libraries(camera_ros
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
```

- [ ] **Step 5: 提交构建配置改动**

```bash
git add src/ucar_camera/package.xml src/ucar_camera/CMakeLists.txt
git commit -m "build(ucar_camera): add sensor_msgs + OpenCV deps and camera_ros target"
```

---

## Task 3: 实现 camera_ros.cpp

**Files:**
- Modify (rewrite): [src/ucar_camera/src/camera_ros.cpp](src/ucar_camera/src/camera_ros.cpp)

- [ ] **Step 1: 把 camera_ros.cpp 完整内容覆写为以下代码**

```cpp
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <stdexcept>
#include <string>

#include "ucar_camera.h"

namespace
{
class CameraRosNode
{
public:
    CameraRosNode()
    {
        int camera_index = pnh_.param<int>("camera_index", 0);
        int image_width = pnh_.param<int>("image_width", 640);
        int image_height = pnh_.param<int>("image_height", 480);
        std::string topic = pnh_.param<std::string>("cam_topic_name", "/ucar_camera/image_raw");
        frame_id_ = pnh_.param<std::string>("frame_id", "opencv");
        int rate = pnh_.param<int>("rate", 30);
        enable_undistort_ = pnh_.param<bool>("enable_undistort", true);
        rate_hz_ = static_cast<double>(rate);

        cam_ = std::make_unique<CameraCapture>(camera_index, image_width, image_height);
        img_pub_ = nh_.advertise<sensor_msgs::Image>(topic, 1);

        ROS_INFO("camera_ros started: index=%d %dx%d topic=%s rate=%.1f undistort=%d",
                 camera_index, image_width, image_height, topic.c_str(), rate_hz_, enable_undistort_ ? 1 : 0);
    }

    void run()
    {
        ros::Rate rate(rate_hz_);
        sensor_msgs::Image img_msg;
        while (ros::ok())
        {
            cv::Mat frame = cam_->captureFrame();
            if (frame.empty())
            {
                ROS_WARN_THROTTLE(1.0, "camera capture returned empty frame");
                rate.sleep();
                continue;
            }
            if (enable_undistort_)
            {
                frame = cam_->correctFrame(frame);
            }
            cv::flip(frame, frame, 1);

            img_msg.header.stamp = ros::Time::now();
            img_msg.header.frame_id = frame_id_;
            img_msg.height = frame.rows;
            img_msg.width = frame.cols;
            img_msg.encoding = "bgr8";
            img_msg.is_bigendian = 0;
            img_msg.step = frame.cols * 3;
            img_msg.data.assign(frame.datastart, frame.dataend);

            img_pub_.publish(img_msg);
            rate.sleep();
        }
    }

private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_{"~"};
    ros::Publisher img_pub_;
    std::unique_ptr<CameraCapture> cam_;
    std::string frame_id_;
    bool enable_undistort_ = true;
    double rate_hz_ = 30.0;
};
}  // namespace

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ucar_camera");
    try
    {
        CameraRosNode node;
        node.run();
    }
    catch (const std::exception &e)
    {
        ROS_ERROR("ucar_camera node failed: %s", e.what());
        return 1;
    }
    return 0;
}
```

要点：
- `pnh_{"~"}` 用成员初始化器，构造函数体内已能用 `pnh_.param<...>` 读私有参数。
- `cam_` 用 `unique_ptr` 延迟构造，便于构造异常抛出时 `main` 捕获。`CameraCapture` 三个参数的构造函数 [ucar_camera.cpp:20-35](src/ucar_camera/src/ucar_camera.cpp#L20-L35) 在打不开相机时抛 `std::runtime_error`，正好被 `main` 的 `try/catch` 接住。
- `ROS_WARN_THROTTLE(1.0, ...)` 对应 Python `rospy.logwarn_throttle(1.0, ...)`。
- `frame.datastart` / `frame.dataend` 在 OpenCV 4 是 `const uchar*`，`std::vector<uint8_t>::assign` 接受迭代器对，能正确拷贝整段连续 BGR 数据。

- [ ] **Step 2: 提交 camera_ros.cpp 实现**

```bash
git add src/ucar_camera/src/camera_ros.cpp
git commit -m "feat(ucar_camera): add camera_ros C++ node publishing sensor_msgs/Image"
```

---

## Task 4: 编译与运行验证

**Files:** 无（验证步骤）

- [ ] **Step 1: colcon 编译 ucar_camera**

Run（按项目实际 workspace 路径替换）：

```bash
cd /path/to/ucar_ws
colcon build --packages-select ucar_camera
```

Expected:
- 无错误退出
- 输出包含 `Finished <<< ucar_camera`
- `install/ucar_camera/lib/ucar_camera/camera_ros` 可执行文件存在

若链接报 `undefined reference to cv::*`，检查 Task 2 Step 4 的 `target_link_libraries` 是否包含 `${OpenCV_LIBRARIES}`。
若编译报 `sensor_msgs/Image.h: No such file or directory`，检查 Task 2 Step 1 的 `find_package` 是否加了 `sensor_msgs`。
若编译报 `ucar_camera.h: No such file or directory`，检查 Task 2 Step 3 的 `include_directories` 是否去掉了 `# include` 的 `#`。

- [ ] **Step 2: source 环境**

```bash
source install/setup.bash
```

- [ ] **Step 3: 启动 roscore 与 camera_ros**

终端 A：

```bash
roscore
```

终端 B（source 后）：

```bash
rosrun ucar_camera camera_ros
```

Expected:
- 终端 B 打印一行 `camera_ros started: index=0 640x480 topic=/ucar_camera/image_raw rate=30.0 undistort=1`
- 无 `Could not open camera` 报错（若报错说明相机索引不对或设备被占）

- [ ] **Step 4: rostopic 验证发布频率**

终端 C：

```bash
rostopic hz /ucar_camera/image_raw
```

Expected: 看到 25~30 Hz 之间稳定的 `average rate`，无 `no new messages` 报错。

- [ ] **Step 5: rostopic 验证 header 字段**

```bash
rostopic echo -n 1 /ucar_camera/image_raw/header
```

Expected:
```yaml
seq: <某个数>
stamp:
  secs: <当前时间秒>
  nsecs: <纳秒>
frame_id: "opencv"
```

- [ ] **Step 6: rqt_image_view 看画面**

```bash
rqt_image_view /ucar_camera/image_raw
```

Expected: 看到相机画面，左右方向与原始场景镜像（因为代码里 `cv::flip(frame, frame, 1)`）。

- [ ] **Step 7: 参数验证 — 关闭去畸变**

Ctrl+C 停掉 camera_ros，重启：

```bash
rosrun ucar_camera camera_ros _enable_undistort:=false
```

Expected: 启动日志末尾变 `undistort=0`；`rqt_image_view` 中画面边缘出现明显桶形畸变（与 Python `_enable_undistort:=False` 一致）。

- [ ] **Step 8: 参数验证 — 降频**

Ctrl+C 后重启：

```bash
rosrun ucar_camera camera_ros _rate:=15
```

Expected: 启动日志显示 `rate=15.0`；`rostopic hz` 实测约 15 Hz。

- [ ] **Step 9: 异常路径 — 空帧警告**

保持相机索引对一个不存在/未插的设备，例如：

```bash
rosrun ucar_camera camera_ros _camera_index:=9
```

Expected:
- `CameraCapture` 构造抛异常 → main 的 catch 打印 `ucar_camera node failed: Could not open camera`
- 进程非零退出（`echo $?` 应非 0）

（构造失败的路径走不到空帧循环；空帧循环的验证需要在相机临时掉线时观察，本步骤主要验证异常退出码。）

---

## 完成标准

全部满足才算完成：

1. Task 1-3 所有步骤已 commit（应有 2 个新 commit：build 配置 + cpp 实现）
2. `colcon build --packages-select ucar_camera` 通过
3. `rosrun ucar_camera camera_ros` 起来后 `rostopic hz /ucar_camera/image_raw` 看到 ~30 Hz
4. `rostopic echo -n 1 /ucar_camera/image_raw/header` 看到 `frame_id: "opencv"`
5. `rqt_image_view` 看到镜像翻转的画面
6. `_enable_undistort:=false` 看到边缘畸变
7. `_camera_index:=9` 走异常路径，进程非零退出
