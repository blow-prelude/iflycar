# TURNING 状态单边线生成 mid_line 设计

## 背景

`find_way.cpp` 与 `find_way_ros.cpp` 在 TURNING 状态下沿用 `calculate_mid_line` 的"左右边线取中"逻辑 `(supple_left + supple_right) / 2`。转弯期间一侧边线经常不可靠或丢失，导致 mid_line 抖动，进而影响 `fit_mid_line_` 和 `track_target_p` 计算出的 `x_error`。

本次改动：进入 TURNING 状态后，由一个持久"模式变量"控制 mid_line 生成方式，默认采用「左边线 + 固定偏移」作为 mid_line。

## 范围

只影响 STRAIGHT_TRACKING / CROSS / TURNING / TRACKING2 这条状态链；`RIGHT_TURNING`（ROS 版为 `RIGHT_TRACKING`）/ `LEFT_TURNING`（`LEFT_TRACKING`）走 task_1 + `fit_polynomial2` 分支，不调用 `calculate_mid_line`，不受影响。

## 数据结构

### 新增枚举（[image_process.h](src/vision_line/include/image_process.h)，`MissLineState` 之后）

```cpp
enum MidLineMode
{
    MID_AVG = 0,       // (supple_left + supple_right) / 2，原有行为
    LEFT_OFFSET = 1,   // supple_left + offset
    RIGHT_OFFSET = 2   // supple_right - offset
};
```

### `ImageProcessConfig` 末尾新增字段

```cpp
// TURNING 状态下用单边线生成 mid_line 的横向偏移（像素）
int turning_mid_offset = 80;
```

### `ImageProcess` 新增成员与接口

```cpp
// public
void set_mid_line_mode(MidLineMode mode) { mid_line_mode_ = mode; }
MidLineMode get_mid_line_mode() const { return mid_line_mode_; }

// private
MidLineMode mid_line_mode_ = LEFT_OFFSET;  // 默认 LEFT_OFFSET
```

## `calculate_mid_line` 改造（[image_process.cpp:1419](src/vision_line/src/image_process.cpp#L1419)）

按 `mid_line_mode_` 分支：

```cpp
void ImageProcess::calculate_mid_line(cv::Mat &img)
{
    int img_h = img.rows;
    int img_w = img.cols;

    fill_boundary(left_line_, right_line_, {img_h, img_w},
                  supple_left_line_, supple_right_line_, true);

    if (mid_line_mode_ == LEFT_OFFSET)
    {
        int n = supple_left_line_.size();
        mid_line_.resize(n);
        for (int i = 0; i < n; i++)
        {
            int x = supple_left_line_[i].x + config_.turning_mid_offset;
            mid_line_[i].x = std::max(0, std::min(x, img_w - 1));
            mid_line_[i].y = supple_left_line_[i].y;
        }
    }
    else if (mid_line_mode_ == RIGHT_OFFSET)
    {
        int n = supple_right_line_.size();
        mid_line_.resize(n);
        for (int i = 0; i < n; i++)
        {
            int x = supple_right_line_[i].x - config_.turning_mid_offset;
            mid_line_[i].x = std::max(0, std::min(x, img_w - 1));
            mid_line_[i].y = supple_right_line_[i].y;
        }
    }
    else // MID_AVG
    {
        int n = std::min(supple_left_line_.size(), supple_right_line_.size());
        mid_line_.resize(n);
        for (int i = 0; i < n; i++)
        {
            mid_line_[i].x = (supple_left_line_[i].x + supple_right_line_[i].x) / 2.0;
            mid_line_[i].y = (supple_left_line_[i].y + supple_right_line_[i].y) / 2.0;
        }
    }

    update_prev_frame_lines();
}
```

`std::max` / `std::min` 已在 image_process.cpp 大量使用（44 处），头文件经 `<opencv2/opencv.hpp>` 间接可用，无需额外 include。用 `std::max(0, std::min(x, img_w-1))` 而非 `std::clamp`，避免 C++17 依赖（项目 CMakeLists.txt 未显式设置 C++ 标准）。

## `find_way.cpp` 状态机改动

### a. CROSS → TURNING（[find_way.cpp:191-195](src/vision_line/src/find_way.cpp#L191-L195)）

```cpp
if (img_process.judge_enter_turning(stop_mid, binary_img.rows, binary_img.cols, y_norm))
{
    state = ProcessState::TURNING;
    img_process.set_mid_line_mode(LEFT_OFFSET);  // 新增
    miss_line = NO_MISS;                          // 新增：进入 TURNING 时重置丢线状态
    std::cout << "state: CROSS -> TURNING at y=" << y_norm << std::endl;
}
```

### b. TURNING → TRACKING2（[find_way.cpp:204-208](src/vision_line/src/find_way.cpp#L204-L208)）

```cpp
if (std::abs(x_error) <= turning_end_x_error_abs_max_)
{
    state = ProcessState::TRACKING2;
    img_process.set_mid_line_mode(MID_AVG);  // 新增
    std::cout << "state: TURNING -> TRACKING2" << std::endl;
}
```

### c. STRAIGHT_TRACKING / CROSS 安全网

在 `calculate_mid_line` 之前（[find_way.cpp:179-181](src/vision_line/src/find_way.cpp#L179-L181) 附近）加防御性复位：

```cpp
if (state == ProcessState::STRAIGHT_TRACKING || state == ProcessState::CROSS)
{
    img_process.set_mid_line_mode(MID_AVG);
}
```

## `find_way_ros.cpp` 状态机改动

### a. CROSS → TURNING（[find_way_ros.cpp:239-243](src/vision_line/src/find_way_ros.cpp#L239-L243)）

```cpp
state_ = TURNING;
ros::param::set(turning_flag_param_, 1);
processor_.set_mid_line_mode(LEFT_OFFSET);  // 新增
miss_line_ = NO_MISS;                        // 新增（兜底）
ROS_INFO("State: CROSS -> TURNING ...");
```

### b. TURNING → TRACKING2（两条路径都要切回）

外部 flag 清零路径（[find_way_ros.cpp:250-254](src/vision_line/src/find_way_ros.cpp#L250-L254)）：

```cpp
if (flag == 0)
{
    state_ = TRACKING2;
    processor_.set_mid_line_mode(MID_AVG);  // 新增
    ROS_INFO("State: TURNING -> TRACKING2 (flag cleared by controller)");
}
```

视觉判定结束路径（[find_way_ros.cpp:272-277](src/vision_line/src/find_way_ros.cpp#L272-L277)）：

```cpp
if (y_pixel >= 0 && std::abs(x_error) <= turning_end_x_error_abs_max_)
{
    state_ = TRACKING2;
    ros::param::set(turning_flag_param_, 0);
    processor_.set_mid_line_mode(MID_AVG);  // 新增
    ROS_INFO("State: TURNING -> TRACKING2 (visual end detected, ...)");
}
```

### c. Reset 兜底（[find_way_ros.cpp:137-144](src/vision_line/src/find_way_ros.cpp#L137-L144)）

```cpp
if (reset_requested_)
{
    state_ = IDLE;
    t0_set = false;
    miss_line_ = NO_MISS;
    processor_.set_mid_line_mode(MID_AVG);  // 新增
    reset_requested_ = false;
    ROS_INFO("State machine reset to IDLE (direction changed)");
}
```

### d. STRAIGHT_TRACKING / CROSS 安全网

在 `get_side_line_task_2 / calculate_mid_line` 之前（[find_way_ros.cpp:228-231](src/vision_line/src/find_way_ros.cpp#L228-L231) 附近）：

```cpp
if (state_ == STRAIGHT_TRACKING || state_ == CROSS)
{
    processor_.set_mid_line_mode(MID_AVG);
}
```

## 边界条件与不变量

| 场景 | 处理 |
|---|---|
| `LEFT_OFFSET` 时 `supple_left_line_` 为空 | `mid_line_` 为空，`fit_polynomial` 内部已做 `size() < 5` 兜底（[image_process.cpp:605](src/vision_line/src/image_process.cpp#L605)），`fit_mid_line_` 也为空 → `track_target_p` 返回 `-1000`，`abs(-1000) > 15` → 不会误进 TRACKING2。安全 |
| 偏移后 x 越界 | `std::max(0, std::min(x, img_w-1))` 限制到 `[0, img_w-1]` |
| `judge_turing_end` | **不受影响**。直接读 `left_line_.back()` / `right_line_.back()`（原始检测线），见 [image_process.cpp:351-353](src/vision_line/src/image_process.cpp#L351-L353) |
| `fit_polynomial` | **不受影响**。对 `mid_line_` 分段拟合，`left+offset` 输入也能正常工作 |
| `draw_line` 可视化 | `mid_line_`（蓝）与 `fit_mid_line_`（白）会显示在 `left+offset` 位置，便于调试 |
| `TRACKING2` 状态 | mode 已切回 `MID_AVG`，恢复 `(left+right)/2` |
| `RIGHT_TURNING` / `LEFT_TURNING`（ROS 版 `RIGHT_TRACKING` / `LEFT_TRACKING`） | 走 task_1 + `fit_polynomial2` 分支，不调用 `calculate_mid_line`，不受影响 |

## 验证方式

C++ + OpenCV 项目无单测，用人工验证清单：

1. **编译通过**：`cd build && colcon build --packages-select vision_line`（或现有构建命令）
2. **直行（STRAIGHT_TRACKING）**：mid_line 蓝点仍穿过画面中央，与原行为一致
3. **看到停止线进入 CROSS**：mid_line 不变
4. **CROSS → TURNING**：控制台打印 `state: CROSS -> TURNING`，画布上 mid_line / fit_mid_line 整体向右平移 80 像素（`left.x + 80`）
5. **TURNING 期间**：mid_line 跟着左边线走，x 始终 ≈ `left.x + 80`
6. **TURNING → TRACKING2**：mid_line 回到 `(left+right)/2`
7. **改 mode 验证**：把成员默认值改成 `RIGHT_OFFSET` 重新编译，观察镜像效果
8. **调节偏移**：在 main 里改 `config.turning_mid_offset = 100;`，重新跑，观察平移距离变化
9. **ROS 版额外验证**：通过 `/vision_line_direction_out` 发送方向切换触发 reset，确认 mode 复位到 `MID_AVG`；通过外部清零 `turning_flag_param_` 触发 TURNING → TRACKING2，确认 mode 也复位

## 改动清单

- [image_process.h](src/vision_line/include/image_process.h)：新增 `MidLineMode` 枚举、`ImageProcessConfig::turning_mid_offset`、`ImageProcess::mid_line_mode_` 成员 + getter/setter
- [image_process.cpp](src/vision_line/src/image_process.cpp)：`calculate_mid_line` 分支化（无需额外 include，`std::max`/`std::min` 已通过现有头文件间接可用）
- [find_way.cpp](src/vision_line/src/find_way.cpp)：CROSS → TURNING 切 LEFT_OFFSET 并重置 `miss_line`；TURNING → TRACKING2 切 MID_AVG；STRAIGHT/CROSS 前置防御性复位
- [find_way_ros.cpp](src/vision_line/src/find_way_ros.cpp)：同上三处 + reset 兜底切 MID_AVG

不动 `judge_turing_end` / `fit_polynomial` / `fit_polynomial2` / `draw_line` / `get_side_line_task_*` / `get_stop_line` / `fill_boundary`。
