# ucar_camera C++ ROS 节点设计

## 背景

当前相机采集只在 [ucar_camera.py](src/ucar_camera/scripts/ucar_camera.py) 一个 Python 节点里实现：开 `cv2.VideoCapture`、可选去畸变、水平翻转、手动填 `sensor_msgs/Image` 字段后发布到 `/ucar_camera/image_raw`。

C++ 侧已有 [CameraCapture](src/ucar_camera/include/ucar_camera.h) 类封装了 `cv::VideoCapture`、采集 (`captureFrame`)、去畸变 (`correctFrame`)，但还没有一个对应的 ROS 节点把它接出来。[camera_ros.cpp](src/ucar_camera/src/camera_ros.cpp) 目前是空文件，[CMakeLists.txt](src/ucar_camera/CMakeLists.txt) 也是默认模板（没有任何 target），[package.xml](src/ucar_camera/package.xml) 缺 `sensor_msgs` 和 OpenCV 依赖。

本次改动：仿照 Python 节点，在 `camera_ros.cpp` 里实现一个 C++ ROS 节点，调用 `CameraCapture` 的相关函数完成采集→去畸变→翻转→发布；同时把 CMake / package.xml 接好，使其能 `rosrun ucar_camera camera_ros` 跑起来。

## 范围

只动 `ucar_camera` 包内三个文件：`camera_ros.cpp`（新建内容）、`CMakeLists.txt`（加 target 和依赖）、`package.xml`（加依赖声明）。不修改 `ucar_camera.h` / `ucar_camera.cpp`（`CameraCapture` 类保持原样）。

## 数据结构

不新增类、不修改 `CameraCapture`。`camera_ros.cpp` 内部定义一个 file-local 的 `CameraRosNode` 类（放匿名命名空间里，避免链接符号冲突），成员：

```cpp
class CameraRosNode
{
public:
    CameraRosNode();          // 读取参数、构造 CameraCapture、建 publisher
    void run();                // 主循环
private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;      // 私有命名空间，读 ~ 参数
    ros::Publisher img_pub_;
    std::unique_ptr<CameraCapture> cam_;
    std::string frame_id_;
    bool enable_undistort_;
    double rate_hz_;
};
```

`main()` 里 `ros::init` 后构造节点，异常捕获后 `ROS_ERROR` 并返回非零。

## 参数

全部用私有参数（`~` 前缀），命名与 Python 保持一致或语义对齐：

| 参数名 | 类型 | 默认值 | 对应 Python | 说明 |
|---|---|---|---|---|
| `~camera_index` | int | `0` | `device_path` | C++ 类只收 int，用索引代替路径 |
| `~image_width` | int | `640` | `~image_width` | 透传给 `CameraCapture` 构造 |
| `~image_height` | int | `480` | `~image_height` | 透传给 `CameraCapture` 构造 |
| `~cam_topic_name` | string | `"/ucar_camera/image_raw"` | `~cam_topic_name` | 默认带前导 `/`，话题名解析为绝对 |
| `~frame_id` | string | `"opencv"` | 硬编码 `"opencv"` | Python 硬编码；C++ 暴露为参数更顺手 |
| `~rate` | int | `30` | `~rate` | 循环频率 Hz |
| `~enable_undistort` | bool | `true` | `~enable_undistort` | 控制是否调用 `correctFrame` |

**舍弃** Python 里的：
- `~calib_width` / `~calib_height` / `~undistort_alpha`：Python 代码里读完就丢，从未实际使用
- `~fourcc`：`CameraCapture` 类没暴露设置 `CAP_PROP_FOURCC` 的接口，要支持得改类，超出本次范围
- `device_path`：按用户决定改用 `camera_index`

`CameraCapture` 第三个构造函数（带 `mtx` / `dist`）不需要用——默认构造已经把与 Python 相同的标定参数写死在头文件里了。

## 主循环（`CameraRosNode::run`）

```cpp
void CameraRosNode::run()
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
            frame = cam_->correctFrame(frame);  // 内部失败时返回原图，安全
        }
        cv::flip(frame, frame, 1);  // 水平翻转，与 Python 一致

        img_msg.header.stamp    = ros::Time::now();
        img_msg.header.frame_id = frame_id_;
        img_msg.height          = frame.rows;
        img_msg.width           = frame.cols;
        img_msg.encoding        = "bgr8";
        img_msg.is_bigendian    = 0;
        img_msg.step            = frame.cols * 3;
        img_msg.data.assign(frame.datastart, frame.dataend);

        img_pub_.publish(img_msg);
        rate.sleep();
    }
}
```

要点：
- **不用 cv_bridge**，按用户决定手动填字段。`cv::Mat::datastart` / `dataend` 保证连续内存拷贝（`captureFrame` 和 `correctFrame` 返回的 Mat 默认连续；`flip` 原地操作若涉及非连续需注意，但 3 通道 BGR Mat 的水平 flip 输出仍连续）。
- **`ROS_WARN_THROTTLE(1.0, ...)`** 对应 Python 的 `rospy.logwarn_throttle(1.0, ...)`，1 秒最多一条。
- **`ros::ok()`** 作为循环退出条件，对应 `rospy.is_shutdown()`。

## 错误处理

| 场景 | 处理 |
|---|---|
| `CameraCapture` 构造失败（相机打不开） | 构造函数抛 `std::runtime_error`，`main()` 里 `try/catch` 捕获 → `ROS_ERROR` 打印 → `return 1` |
| `captureFrame()` 返回空 Mat | `ROS_WARN_THROTTLE` 后 `continue`，等下一拍 |
| `correctFrame()` 内部异常 | 类内已 `try/catch`，失败时返回原帧（见 [ucar_camera.cpp:97-101](src/ucar_camera/src/ucar_camera.cpp#L97-L101)），节点无感 |
| ROS shutdown（Ctrl+C） | `ros::ok()` 变 false，循环退出，析构 `CameraCapture` 自动 `cap.release()` |

## CMakeLists.txt 改动

当前 [CMakeLists.txt](src/ucar_camera/CMakeLists.txt) 的 `find_package` 只声明了 `roscpp / rospy / std_msgs`，且没有任何 `add_executable`。需要：

1. `find_package` 增加 `sensor_msgs`：

```cmake
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
)
```

2. 新增 OpenCV 查找：

```cmake
find_package(OpenCV REQUIRED)
```

3. `include_directories` 启用 `include/` 并加 OpenCV：

```cmake
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
```

4. 在 `## Build ##` 段新增 executable（直接把 `ucar_camera.cpp` 一起编进可执行，不单独做 library）：

```cmake
add_executable(camera_ros
  src/camera_ros.cpp
  src/ucar_camera.cpp
)
target_link_libraries(camera_ros
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
```

## package.xml 改动

按文件现有风格（`build_depend` / `build_export_depend` / `exec_depend` 三联），在对应分组各加一行：

```xml
<build_depend>sensor_msgs</build_depend>
<build_depend>libopencv-dev</build_depend>

<build_export_depend>sensor_msgs</build_export_depend>
<build_export_depend>libopencv-dev</build_export_depend>

<exec_depend>sensor_msgs</exec_depend>
<exec_depend>libopencv-dev</exec_depend>
```

`libopencv-dev` 是 Debian/Ubuntu 系统包名，在 ROS Melodic/Noetic 上是标准做法。

## 验证方式

无单测，按人工清单：

1. **编译通过**：`cd <ws> && colcon build --packages-select ucar_camera`（或现有构建命令），无报错，`install/` 下出现 `camera_ros` 可执行
2. **运行**：`roscore` & `rosrun ucar_camera camera_ros`，节点起来无异常
3. **话题**：`rostopic hz /ucar_camera/image_raw` 接近 30 Hz；`rostopic echo -n 1 /ucar_camera/image_raw/header` 看到 `frame_id: "opencv"` 和当前时间戳
4. **图像正确**：`rqt_image_view /ucar_camera/image_raw` 看到画面，且左右翻转与 Python 节点一致
5. **去畸变开关**：`rosrun ucar_camera camera_ros _enable_undistort:=false`，对比图像边缘有明显桶形畸变（与 Python 同参数一致）
6. **参数透传**：`_camera_index:=1 _image_width:=1280 _image_height:=720 _rate:=15` 分别验证索引切换、分辨率、频率
7. **空帧**：拔掉相机或对准故障设备，`ROS_WARN` 1 秒一条，节点不崩
8. **Ctrl+C**：`ros::ok()` 退出循环，进程正常结束，无残留

## 改动清单

- [src/ucar_camera/src/camera_ros.cpp](src/ucar_camera/src/camera_ros.cpp)：从空文件实现 `CameraRosNode` + `main`（约 80 行）
- [src/ucar_camera/CMakeLists.txt](src/ucar_camera/CMakeLists.txt)：`find_package` 加 `sensor_msgs`；新增 `find_package(OpenCV REQUIRED)`；`include_directories` 加 `include` 和 OpenCV；新增 `camera_ros` target
- [src/ucar_camera/package.xml](src/ucar_camera/package.xml)：三联依赖各加 `sensor_msgs` 和 `libopencv-dev`

不动 [ucar_camera.h](src/ucar_camera/include/ucar_camera.h) / [ucar_camera.cpp](src/ucar_camera/src/ucar_camera.cpp) / [scripts/ucar_camera.py](src/ucar_camera/scripts/ucar_camera.py)。
