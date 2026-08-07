# STRAIGHT_TRACKING 左边线单边跟踪

## 背景

当前 `find_way.cpp` 的 `STRAIGHT_TRACKING` 状态调用 `get_side_line_task_2(BOTH)` 搜索左右两条边线，再 `calculate_mid_line(MID_AVG)` 取平均，最后 `fit_polynomial` 拟合得到 `fit_mid_line_`。

实际场景里直道经常只能稳定看到左边线（右边丢失或抖动），平均之后中线被拉向图像边缘，跟踪误差变大。希望加一个开关：在 `STRAIGHT_TRACKING` 状态下，默认只搜左边线、按左边线 + 横向偏移得到 `fit_mid_line_`。

## 目标

- 新增一个布尔变量控制 `STRAIGHT_TRACKING` 是否走"单左边线"路径，默认开启。
- 跳过右边线的逐行搜索（真正的性能优化，不是"搜了再扔"）。
- 复用现有 `LEFT_OFFSET` 模式 + `fit_polynomial`，不增加新的 `mid_line_` 计算路径。
- 不动 TURNING（`RIGHT_TURNING` / `LEFT_TURNING`）走 `task_1` + `fit_polynomial2` 的代码路径。

## 非目标

- 不改 `turning_mid_offset` 配置（用户明确说当前逻辑后续要推翻，暂时复用）。
- 不暴露 `supple_left_line_` 等私有成员。
- 不重构现有 `get_side_line_task_2` 的逐行搜索主循环结构。

## 涉及文件

| 文件 | 改动 |
|------|------|
| [src/vision_line/include/image_process.h](../../../src/vision_line/include/image_process.h) | 新增 `SearchSide` enum；`get_side_line_task_2` 增加 `SearchSide side = BOTH` 参数 |
| [src/vision_line/src/image_process.cpp](../../../src/vision_line/src/image_process.cpp) | 在 `get_side_line_task_2` 内根据 `side` 跳过对应一侧的搜索 |
| [src/vision_line/src/find_way.cpp](../../../src/vision_line/src/find_way.cpp) | 新增 `bool straight_left_only = true;`；`else` 分支按 flag 与 state 选 `side` 和 `mode` |

## 详细设计

### 1. `image_process.h`

在 `MidLineMode` 附近新增 enum：

```cpp
enum SearchSide
{
    BOTH = 0,
    LEFT_ONLY = 1,
    RIGHT_ONLY = 2,
};
```

修改 `get_side_line_task_2` 声明（带默认值，向后兼容现有调用方）：

```cpp
void get_side_line_task_2(cv::Mat &img, cv::Mat &canvas, bool is_draw,
                          bool find_corner, SearchSide side = BOTH);
```

### 2. `image_process.cpp`

在 `get_side_line_task_2` ([image_process.cpp:1174](../../../src/vision_line/src/image_process.cpp#L1174)) 内：

- 把"搜索左边线"代码段用 `if (side != RIGHT_ONLY)` 包起来。
- 把"搜索右边线"代码段用 `if (side != LEFT_ONLY)` 包起来。
- 被跳过一侧的 `left_line_` / `right_line_` 保持为空（`clear_lines()` 已清空过）。

后续流程无需改动：`fill_boundary` ([image_process.cpp:509](../../../src/vision_line/src/image_process.cpp#L509)) 在 line 564-579 已经处理了"右线丢失、左线存在"的情况——以左线为基准插值得到 `supple_left_line_`，并按左线的 y 序列造一个 x = img_w-1 的占位 `supple_right_line_`。`calculate_mid_line` 在 `LEFT_OFFSET` 模式下只读 `supple_left_line_`，所以占位的右线不影响结果。

### 3. `find_way.cpp`

在 [find_way.cpp:77-78](../../../src/vision_line/src/find_way.cpp#L77-L78) 参数区追加：

```cpp
bool straight_left_only = true; // STRAIGHT_TRACKING 是否只走左边线
```

把 [find_way.cpp:156-171](../../../src/vision_line/src/find_way.cpp#L156-L171) 的 `else` 分支替换为：

```cpp
else
{
    SearchSide side = BOTH;
    MidLineMode mode = MID_AVG;
    if (state == ProcessState::STRAIGHT_TRACKING && straight_left_only)
    {
        side = LEFT_ONLY;
        mode = LEFT_OFFSET;
    }
    img_process.set_mid_line_mode(mode);

    cv::Mat canvas = img_process.return_frame();
    img_process.get_side_line_task_2(binary_img, canvas, true, false, side);
    img_process.calculate_mid_line(binary_img);
    img_process.fit_polynomial();

    img_process.draw_line(canvas, fps, state_name(state));

    cv::imshow("binary", binary_img);
    cv::imshow("canvas", canvas);
}
```

原 `else` 分支里那个空的 `if (state == ProcessState::STRAIGHT_TRACKING) {}` 占位块在替换中被删掉，逻辑吸收到 side/mode 的条件计算里。

## 数据流

```
STRAIGHT_TRACKING (straight_left_only = true):
  set_mid_line_mode(LEFT_OFFSET)
  get_side_line_task_2(LEFT_ONLY)  → left_line_ 有内容, right_line_ 空
  calculate_mid_line               → fill_boundary 走"右线丢失"分支
                                    → supple_left_line_ 插值完整
                                    → mid_line_ = supple_left + turning_mid_offset (默认 40)
  fit_polynomial                   → fit_mid_line_

非转弯状态 (straight_left_only = false 或其他):
  set_mid_line_mode(MID_AVG)
  get_side_line_task_2(BOTH)       → 两边都搜
  calculate_mid_line               → mid_line_ = (L+R)/2
  fit_polynomial                   → fit_mid_line_

RIGHT_TURNING / LEFT_TURNING:
  不受影响, 走 task_1 + fit_polynomial2
```

## 兼容性

- `get_side_line_task_2` 新参数有默认值 `BOTH`，C++ 侧另一个调用方 [find_way_ros.cpp:225](../../../src/vision_line/src/find_way_ros.cpp#L225) 无需修改即可继续按两边搜索行为运行（Python 端 `image_process.py` / `image_process_ros.py` 是另一套实现，不在本次范围内）。
- `straight_left_only` 默认为 `true`，符合"默认只走左边线"的需求；测试两边搜索时手动置 `false`。
- `turning_mid_offset` 被 STRAIGHT_TRACKING 和 TURNING 共用，TURNING 路径已经不动，耦合在可接受范围内。

## 风险

- `LEFT_ONLY` 模式下如果左边线本身丢线（`left_line_` 也空），`fill_boundary` 走 line 582-600 的"两边都丢线"分支，回退到上一帧。多帧连续丢线时 mid_line 不更新，fit 会失败打日志——但这是现有行为，不是本次引入的退化。
- 后续如果真要推翻 `turning_mid_offset` 共用，需要同步检查 TURNING 的 offset 来源。
