# TURNING 单边线 mid_line 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 TURNING 状态下让 `calculate_mid_line` 用「左边线 + 偏移」生成 mid_line，由 `mid_line_mode_` 成员控制，默认 `LEFT_OFFSET`。

**Architecture:** 给 `ImageProcess` 加一个 mode 成员与 setter；`calculate_mid_line` 按 mode 分支（`MID_AVG` 原行为 / `LEFT_OFFSET` / `RIGHT_OFFSET`）。两个状态机入口（`find_way.cpp` 与 `find_way_ros.cpp`）在 TURNING 边界处切 mode。

**Tech Stack:** C++14、OpenCV、ROS 1（catkin_make）、cv_bridge。

**Spec:** [docs/superpowers/specs/2026-07-15-turning-midline-offset-design.md](../specs/2026-07-15-turning-midline-offset-design.md)

**构建命令：** 在工作区根目录 `d:/programs/ucar_ws/` 执行 `catkin_make -DCATKIN_WHITELIST_PACKAGES="vision_line"`（或 `catkin_make --pkg vision_line`）。

---

### Task 1: image_process.h — 新增 MidLineMode 枚举与配置

**Files:**
- Modify: [src/vision_line/include/image_process.h](src/vision_line/include/image_process.h)
  - `MissLineState` 枚举之后（约 67 行）
  - `ImageProcessConfig` 结构体末尾（约 58 行）
  - `ImageProcess` 类 public 区（约 112 行附近，靠近 `get_fit_mid_line`）
  - `ImageProcess` 类 private 区（约 116 行附近，`config_` 之前）

- [ ] **Step 1: 在 `MissLineState` 枚举后新增 `MidLineMode` 枚举**

在 [image_process.h:67](src/vision_line/include/image_process.h#L67)（`MissLineState` 枚举的 `};` 之后，`struct LineFit` 之前）插入：

```cpp
// mid_line 计算模式
enum MidLineMode
{
    MID_AVG = 0,       // (supple_left + supple_right) / 2
    LEFT_OFFSET = 1,   // supple_left + offset
    RIGHT_OFFSET = 2   // supple_right - offset
};
```

- [ ] **Step 2: 在 `ImageProcessConfig` 末尾新增 `turning_mid_offset`**

在 [image_process.h:58](src/vision_line/include/image_process.h#L58)（`tracking2_target_p_index` 之后，`};` 之前）插入：

```cpp
    // TURNING 状态下用单边线生成 mid_line 的横向偏移（像素）
    int turning_mid_offset = 80;
```

- [ ] **Step 3: 在 `ImageProcess` public 区新增 setter/getter**

在 [image_process.h:112](src/vision_line/include/image_process.h#L112)（`get_fit_mid_line()` 那一行之前或之后均可，建议之前）插入：

```cpp
    void set_mid_line_mode(MidLineMode mode) { mid_line_mode_ = mode; }
    MidLineMode get_mid_line_mode() const { return mid_line_mode_; }
```

- [ ] **Step 4: 在 `ImageProcess` private 区新增成员 `mid_line_mode_`**

在 [image_process.h:117](src/vision_line/include/image_process.h#L117)（`private:` 之后，`ImageProcessConfig config_;` 之前）插入：

```cpp
    MidLineMode mid_line_mode_ = LEFT_OFFSET; // TURNING 时单边线 mid_line 模式
```

- [ ] **Step 5: 编译验证**

在工作区根目录执行：
```bash
cd d:/programs/ucar_ws && catkin_make -DCATKIN_WHITELIST_PACKAGES="vision_line"
```
期望：编译成功（虽然 `mid_line_mode_` 还没被使用，但头文件能正常解析）。

- [ ] **Step 6: 提交**

```bash
git add src/vision_line/include/image_process.h
git commit -m "feat(vision_line): add MidLineMode enum and turning_mid_offset config"
```

---

### Task 2: image_process.cpp — `calculate_mid_line` 分支化

**Files:**
- Modify: [src/vision_line/src/image_process.cpp:1419-1437](src/vision_line/src/image_process.cpp#L1419-L1437)

- [ ] **Step 1: 替换 `calculate_mid_line` 函数体**

把 [image_process.cpp:1419-1437](src/vision_line/src/image_process.cpp#L1419-L1437) 的函数整体替换为：

```cpp
void ImageProcess::calculate_mid_line(cv::Mat &img)
{
    int img_h = img.rows;
    int img_w = img.cols;

    // 插值+填充边线（同时处理边线情况）
    fill_boundary(this->left_line_, this->right_line_, {img_h, img_w}, this->supple_left_line_, this->supple_right_line_, true);

    if (this->mid_line_mode_ == LEFT_OFFSET)
    {
        // 用左边线 + 偏移
        int n = this->supple_left_line_.size();
        this->mid_line_.resize(n);
        for (int i = 0; i < n; i++)
        {
            int x = this->supple_left_line_[i].x + this->config_.turning_mid_offset;
            this->mid_line_[i].x = std::max(0, std::min(x, img_w - 1));
            this->mid_line_[i].y = this->supple_left_line_[i].y;
        }
    }
    else if (this->mid_line_mode_ == RIGHT_OFFSET)
    {
        // 用右边线 - 偏移
        int n = this->supple_right_line_.size();
        this->mid_line_.resize(n);
        for (int i = 0; i < n; i++)
        {
            int x = this->supple_right_line_[i].x - this->config_.turning_mid_offset;
            this->mid_line_[i].x = std::max(0, std::min(x, img_w - 1));
            this->mid_line_[i].y = this->supple_right_line_[i].y;
        }
    }
    else // MID_AVG
    {
        // 使用优化后的边线计算中线
        int n = std::min(this->supple_left_line_.size(), this->supple_right_line_.size());
        this->mid_line_.resize(n);
        for (int i = 0; i < n; i++)
        {
            this->mid_line_[i].x = (this->supple_left_line_[i].x + this->supple_right_line_[i].x) / 2.0;
            this->mid_line_[i].y = (this->supple_left_line_[i].y + this->supple_right_line_[i].y) / 2.0;
        }
    }

    this->update_prev_frame_lines();
}
```

要点：
- `fill_boundary` / `update_prev_frame_lines` 调用保持不变
- `std::max` / `std::min` 已在文件中大量使用（不需要新增 include）
- 三种 mode 分支，单边模式不取两边线的 min 长度

- [ ] **Step 2: 编译验证**

```bash
cd d:/programs/ucar_ws && catkin_make -DCATKIN_WHITELIST_PACKAGES="vision_line"
```
期望：编译成功，无 warning。

- [ ] **Step 3: 提交**

```bash
git add src/vision_line/src/image_process.cpp
git commit -m "feat(vision_line): branch calculate_mid_line by MidLineMode"
```

---

### Task 3: find_way.cpp — 状态机切换 mode

**Files:**
- Modify: [src/vision_line/src/find_way.cpp](src/vision_line/src/find_way.cpp)
  - 约 179-181 行（STRAIGHT_TRACKING/CROSS 安全网）
  - 约 191-195 行（CROSS → TURNING）
  - 约 204-208 行（TURNING → TRACKING2）

- [ ] **Step 1: 添加 STRAIGHT_TRACKING / CROSS 安全网**

在 [find_way.cpp:179](src/vision_line/src/find_way.cpp#L179) 之前（`cv::Mat canvas = img_process.return_frame();` 之前），插入：

```cpp
                    // 非 TURNING 状态强制使用 MID_AVG，防止 mode 残留
                    if (state == ProcessState::STRAIGHT_TRACKING || state == ProcessState::CROSS)
                    {
                        img_process.set_mid_line_mode(MID_AVG);
                    }
```

注意缩进对齐 `else` 块内的层级（8 空格 + 嵌套）。

- [ ] **Step 2: CROSS → TURNING 处切 LEFT_OFFSET 并重置 miss_line**

把 [find_way.cpp:191-195](src/vision_line/src/find_way.cpp#L191-L195)：

```cpp
                        if (img_process.judge_enter_turning(stop_mid, binary_img.rows, binary_img.cols, y_norm))
                        {
                            state = ProcessState::TURNING;
                            std::cout << "state: CROSS -> TURNING at y=" << y_norm << std::endl;
                        }
```

替换为：

```cpp
                        if (img_process.judge_enter_turning(stop_mid, binary_img.rows, binary_img.cols, y_norm))
                        {
                            state = ProcessState::TURNING;
                            img_process.set_mid_line_mode(LEFT_OFFSET);
                            miss_line = NO_MISS;
                            std::cout << "state: CROSS -> TURNING at y=" << y_norm << std::endl;
                        }
```

- [ ] **Step 3: TURNING → TRACKING2 处切回 MID_AVG**

把 [find_way.cpp:204-208](src/vision_line/src/find_way.cpp#L204-L208)：

```cpp
                            if (std::abs(x_error) <= turning_end_x_error_abs_max_)
                            {
                                state = ProcessState::TRACKING2;
                                std::cout << "state: TURNING -> TRACKING2" << std::endl;
                            }
```

替换为：

```cpp
                            if (std::abs(x_error) <= turning_end_x_error_abs_max_)
                            {
                                state = ProcessState::TRACKING2;
                                img_process.set_mid_line_mode(MID_AVG);
                                std::cout << "state: TURNING -> TRACKING2" << std::endl;
                            }
```

- [ ] **Step 4: 编译验证**

```bash
cd d:/programs/ucar_ws && catkin_make -DCATKIN_WHITELIST_PACKAGES="vision_line"
```
期望：编译成功，`find_way` 可执行文件生成。

- [ ] **Step 5: 提交**

```bash
git add src/vision_line/src/find_way.cpp
git commit -m "feat(vision_line): switch MidLineMode at TURNING boundaries in find_way"
```

---

### Task 4: find_way_ros.cpp — 状态机切换 mode（含 reset 兜底）

**Files:**
- Modify: [src/vision_line/src/find_way_ros.cpp](src/vision_line/src/find_way_ros.cpp)
  - 约 137-144 行（reset 兜底）
  - 约 228-231 行（STRAIGHT_TRACKING/CROSS 安全网）
  - 约 239-243 行（CROSS → TURNING）
  - 约 250-254 行（TURNING → TRACKING2 flag 路径）
  - 约 272-277 行（TURNING → TRACKING2 视觉路径）

- [ ] **Step 1: reset 路径兜底切回 MID_AVG**

把 [find_way_ros.cpp:137-144](src/vision_line/src/find_way_ros.cpp#L137-L144)：

```cpp
                if (reset_requested_)
                {
                    state_ = IDLE;
                    t0_set = false;
                    miss_line_ = NO_MISS;
                    reset_requested_ = false;
                    ROS_INFO("State machine reset to IDLE (direction changed)");
                }
```

替换为：

```cpp
                if (reset_requested_)
                {
                    state_ = IDLE;
                    t0_set = false;
                    miss_line_ = NO_MISS;
                    processor_.set_mid_line_mode(MID_AVG);
                    reset_requested_ = false;
                    ROS_INFO("State machine reset to IDLE (direction changed)");
                }
```

- [ ] **Step 2: STRAIGHT_TRACKING / CROSS 安全网**

在 [find_way_ros.cpp:228](src/vision_line/src/find_way_ros.cpp#L228) 之前（`cv::Mat canvas = processor_.return_frame();` 之前），插入：

```cpp
                    // 非 TURNING 状态强制使用 MID_AVG，防止 mode 残留
                    if (state_ == STRAIGHT_TRACKING || state_ == CROSS)
                    {
                        processor_.set_mid_line_mode(MID_AVG);
                    }
```

- [ ] **Step 3: CROSS → TURNING 处切 LEFT_OFFSET 并重置 miss_line_**

把 [find_way_ros.cpp:237-243](src/vision_line/src/find_way_ros.cpp#L237-L243)：

```cpp
                        if (processor_.judge_enter_turning(stop_mid, proc_h, proc_w, y_norm))
                        {
                            state_ = TURNING;
                            ros::param::set(turning_flag_param_, 1);

                            ROS_INFO("State: CROSS -> TURNING (stop line, y_norm=%.2f)", y_norm);
                        }
```

替换为：

```cpp
                        if (processor_.judge_enter_turning(stop_mid, proc_h, proc_w, y_norm))
                        {
                            state_ = TURNING;
                            ros::param::set(turning_flag_param_, 1);
                            processor_.set_mid_line_mode(LEFT_OFFSET);
                            miss_line_ = NO_MISS;

                            ROS_INFO("State: CROSS -> TURNING (stop line, y_norm=%.2f)", y_norm);
                        }
```

- [ ] **Step 4: TURNING → TRACKING2（外部 flag 清零路径）切回 MID_AVG**

把 [find_way_ros.cpp:250-254](src/vision_line/src/find_way_ros.cpp#L250-L254)：

```cpp
                        if (flag == 0)
                        {
                            state_ = TRACKING2;
                            ROS_INFO("State: TURNING -> TRACKING2 (flag cleared by controller)");
                        }
```

替换为：

```cpp
                        if (flag == 0)
                        {
                            state_ = TRACKING2;
                            processor_.set_mid_line_mode(MID_AVG);
                            ROS_INFO("State: TURNING -> TRACKING2 (flag cleared by controller)");
                        }
```

- [ ] **Step 5: TURNING → TRACKING2（视觉结束路径）切回 MID_AVG**

把 [find_way_ros.cpp:272-277](src/vision_line/src/find_way_ros.cpp#L272-L277)：

```cpp
                            if (y_pixel >= 0 && std::abs(x_error) <= turning_end_x_error_abs_max_)
                            {
                                state_ = TRACKING2;
                                ros::param::set(turning_flag_param_, 0);
                                ROS_INFO("State: TURNING -> TRACKING2 (visual end detected, x_error=%.1f)", x_error);
                            }
```

替换为：

```cpp
                            if (y_pixel >= 0 && std::abs(x_error) <= turning_end_x_error_abs_max_)
                            {
                                state_ = TRACKING2;
                                ros::param::set(turning_flag_param_, 0);
                                processor_.set_mid_line_mode(MID_AVG);
                                ROS_INFO("State: TURNING -> TRACKING2 (visual end detected, x_error=%.1f)", x_error);
                            }
```

- [ ] **Step 6: 编译验证**

```bash
cd d:/programs/ucar_ws && catkin_make -DCATKIN_WHITELIST_PACKAGES="vision_line"
```
期望：编译成功，`find_way_ros` 可执行文件生成。

- [ ] **Step 7: 提交**

```bash
git add src/vision_line/src/find_way_ros.cpp
git commit -m "feat(vision_line): switch MidLineMode at TURNING boundaries in find_way_ros"
```

---

### Task 5: 手动验证（无单测，靠人工跑相机）

**Files:**
- 无修改，仅运行验证

- [ ] **Step 1: 启动 `find_way`，观察直行状态**

```bash
source devel/setup.bash
rosrun vision_line find_way
```

观察画布：
- **STRAIGHT_TRACKING**：mid_line 蓝点穿过画面中央（与改动前一致）
- 控制台每 30 帧打印 FPS

- [ ] **Step 2: 观察进入 CROSS**

让相机看到停止线（或维持直行 3 秒让 `corner_delay_s_` 触发）。
- 控制台打印 `state: STRAIGHT_TRACKING -> CROSS after 3s`
- mid_line 仍然在画面中央，未偏移

- [ ] **Step 3: 观察进入 TURNING**

让相机持续看到停止线且 `y_norm > 0.65`。
- 控制台打印 `state: CROSS -> TURNING at y=...`
- **画布上 mid_line 蓝点整体向右平移 80 像素**（`supple_left.x + 80`）
- fit_mid_line 白点也在偏移后的位置上

- [ ] **Step 4: 观察 TURNING → TRACKING2**

让转弯完成（`judge_turing_end` 触发且 `|x_error| ≤ 15`）。
- 控制台打印 `state: TURNING -> TRACKING2`
- mid_line 回到 `(left+right)/2`（画面中央）

- [ ] **Step 5: 调试切换验证（可选）**

临时把 [image_process.h](src/vision_line/include/image_process.h) 的成员默认改成 `RIGHT_OFFSET`：
```cpp
MidLineMode mid_line_mode_ = RIGHT_OFFSET;
```
重新编译，重跑 Step 3，观察 mid_line 整体向左平移 80 像素（镜像效果）。验证完成后改回 `LEFT_OFFSET`。

- [ ] **Step 6: 偏移量调节验证（可选）**

在 [find_way.cpp:111](src/vision_line/src/find_way.cpp#L111) `ImageProcess img_process(config);` 之前插入：
```cpp
config.turning_mid_offset = 100;
```
重新编译，重跑 Step 3，观察平移距离从 80 变成 100。验证完成后移除测试代码。

- [ ] **Step 7: ROS 版验证（如使用 ROS 部署）**

```bash
source devel/setup.bash
rosrun vision_line find_way_ros
```
在另一终端发布方向消息触发状态机：
```bash
rostopic pub /vision_line_direction_out std_msgs/String "data: 'straight'"
```
观察与 Step 1-4 一致的行为。

额外验证 ROS 版的 reset 路径：发 `'stop'` 再发 `'straight'`，确认 mid_line 在新一轮 STRAIGHT_TRACKING 中回到中央（mode 已被 reset 路径切回 MID_AVG）。

- [ ] **Step 8: 最终提交（如有调试残留代码）**

```bash
git status
# 确认 image_process.h 的默认值仍是 LEFT_OFFSET
# 确认 find_way.cpp 没有 turning_mid_offset = 100 的调试残留
```

如有调试残留，回滚到 Task 4 的提交：
```bash
git checkout src/vision_line/include/image_process.h src/vision_line/src/find_way.cpp
```

---

## 完成标志

- [ ] 4 个 commit 已落地（Task 1/2/3/4 各一个）
- [ ] 两个可执行文件 `find_way` 与 `find_way_ros` 都能正常编译启动
- [ ] 直行 / CROSS 的 mid_line 仍在画面中央
- [ ] TURNING 的 mid_line 偏移到 `left.x + 80`
- [ ] 退出 TURNING 后 mid_line 回到中央
- [ ] ROS 版 reset 后 mode 复位
