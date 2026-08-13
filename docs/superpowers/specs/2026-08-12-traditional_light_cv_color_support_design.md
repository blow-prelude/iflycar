# 传统交通灯亮区约束与候选排序设计

状态：已实现  
适用脚本：`src/traffic_light/scripts/traditional_light_cv.py`

## 1. 背景

当前算法先从 HSV 图像中分别生成绿色、红色和高亮低饱和度掩膜，再对全局亮区做连通域分析。候选通过尺寸、面积、填充率、颜色分数和颜色密度过滤，最终按照 `color_score`、`component_area` 选择一个候选。

在 `1280×960`、`scale=2.0` 的实测场景中，交通灯左右存在白色挡板。水平箭头的高亮区域会与挡板发生粘连，造成候选消失或只留下方向错误的碎片。

问题样本：

- `pictures/capture_1786544471347320238_000720.jpg`：右箭头
- `pictures/capture_1786544476988825876_000839.jpg`：左箭头
- `pictures/capture_1786544480705245656_000920.jpg`：直行箭头

## 2. 现象和根因

### 2.1 左右箭头与背景白色物体粘连

当前亮区规则为：

```text
0 <= H <= 179
0 <= S <= 110
212 <= V <= 255
```

这项规则既会选中白色箭头，也会选中白纸、挡板和其他高亮低饱和度物体。

在保存的左右箭头图片中，箭头和挡板形成了同一个连通域：

```text
bbox=(589, 442, 409, 92)
component_area≈13200
component_fill≈0.35
```

`scale=2.0` 时最大允许宽度为 `180`，因此该连通域因 `width=409` 被淘汰，最终返回 `unknown`。

### 2.2 曝光波动导致错误碎片入选

相机曝光和 JPEG 像素值的轻微变化会改变亮区的连通关系。粘连区域有时会暂时断开，但断开的部分不一定是完整箭头。右箭头日志中的候选：

```text
bbox=(595, 447, 57, 69)
axis=vertical
label=straight
```

该区域是箭头与附近白色物体形成的竖向碎片。PCA 对这个碎片本身的判断没有错误，但输入给 PCA 的并不是完整右箭头。

### 2.3 候选排序容易选择绿色背景附近的碎片

当前最终选择规则为：

```python
max(candidates, key=lambda item: (item.color_score, item.component_area))
```

`color_score` 表示候选外扩窗口中的红色或绿色像素数量，并不直接表示候选形状的完整程度。当多个候选位于绿色灯面附近时，白色挡板碎片可能因为周围绿色像素更多而胜出。

直行日志中 `accepted=2~4`，说明存在多个竞争候选。当前结果虽然是 `straight`，但框的位置会在完整箭头和相邻碎片之间变化。

## 3. 设计目标

1. 切断交通灯箭头与场景中白色物体的亮区连接。
2. 右、左、直行三张问题样本分别稳定输出 `right`、`left`、`straight`。
3. 保留红灯检测能力，不能只使用绿色区域作为约束。
4. 保持既有图片的分类结果不变。
5. 降低多个候选存在时选择稀疏碎片的概率。
6. 新增足够的诊断字段，但保持默认每秒一条的日志节流策略。

## 4. 非目标

- 本次不引入神经网络模型。
- 本次不增加跨帧跟踪或投票。
- 本次不调整相机曝光、白平衡或标定参数。
- 本次不继续降低尺寸、面积和亮度门槛。
- 本次不改变左右、直行的 PCA 与投影分带判定规则。

## 5. 总体方案

```text
BGR 图像
   |
   v
HSV + ROI
   |
   +----> green mask ----+
   |                     |
   +----> red mask ------+--> color mask --> 膨胀 --> color support
   |
   +----> raw bright mask --> 闭运算 --------+--> 按位与 --> supported bright
                                                        |
                                                        v
                                                  连通域候选过滤
                                                        |
                                                        v
                                             完整度加权候选排序
                                                        |
                                                        v
                                                  颜色/方向分类
```

核心思想是：白色箭头应当位于红色或绿色灯面附近，而场景中的大片白色挡板不应整体进入候选集合。

## 6. 详细设计

### 6.1 新增颜色支撑掩膜

先合并红绿颜色掩膜：

```python
color_mask = cv2.bitwise_or(green, red)
```

使用矩形核膨胀，让颜色区域覆盖内部白色箭头：

```python
support_kernel_size = scaled_odd_length(12, scale)
support_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (support_kernel_size, support_kernel_size),
)
color_support = cv2.dilate(color_mask, support_kernel)
```

参数定义：

```python
color_support_kernel_size: int = 12
```

核尺寸必须是奇数：

```python
scaled = max(3, int(round(value * scale)))
if scaled % 2 == 0:
    scaled += 1
```

因此：

- `scale=1.0` 时使用 `13×13`；
- `scale=2.0` 时使用 `25×25`；
- 小分辨率下至少使用 `3×3`。

使用矩形而非椭圆形结构元素是实时性优化。OpenCV 可以用可分离方式执行
矩形膨胀；在 `1280×960` 上，椭圆膨胀实测约 40 ms，矩形膨胀约 3 ms。
基准长度从模拟阶段的 `15` 收紧到 `12`，用于抵消矩形核四角增加的覆盖范围。
三张问题图片和全部现有样本的分类结果一致。十字核也经过测试，但会使既有
直行样本 `01042.jpg` 从 `straight` 回归为 `unknown`，因此未采用。

### 6.2 约束亮区

保留当前亮区阈值和闭运算，然后与颜色支撑掩膜相交：

```python
bright_closed = cv2.morphologyEx(
    bright_raw,
    cv2.MORPH_CLOSE,
    close_kernel,
)
bright_supported = cv2.bitwise_and(bright_closed, color_support)
```

连通域分析必须使用 `bright_supported`，不能继续使用全局 `bright_closed`。

必须使用 `green | red` 生成支撑区。如果只膨胀绿色掩膜，现有红灯样本会失去候选。

### 6.3 保留原始亮区用于诊断

`Masks` 建议同时保留：

```python
bright_raw: np.ndarray
bright: np.ndarray  # color-supported bright mask
color_support: np.ndarray
```

这样可以区分以下两类失败：

- `bright_raw` 中没有箭头：亮度或饱和度阈值问题；
- `bright_raw` 有箭头，但 `bright` 中没有：颜色支撑核或颜色阈值问题。

### 6.4 候选排序加入完整度权重

通过既有过滤规则后，为每个候选计算：

```python
selection_score = color_score * component_fill_density
```

最终排序改为：

```python
max(
    candidates,
    key=lambda item: (
        item.selection_score,
        item.color_score,
        item.component_area,
    ),
)
```

设计理由：

- `color_score` 保证候选确实位于红色或绿色灯面附近；
- `component_fill_density` 惩罚被颜色支撑边界截出的稀疏白色碎片；
- 保留颜色分数和面积作为稳定的次级排序条件。

在直行问题样本中，完整箭头的填充率高于相邻白色碎片。加权后，完整箭头会优先于仅仅拥有更多周边绿色像素的碎片。

### 6.5 不修改方向分类

`classify_green_shape()` 的输入恢复为完整箭头后，现有特征已有较大余量：

- 左：水平轴，投影峰值位于左半部；
- 右：水平轴，投影峰值位于右半部；
- 直行：垂直轴。

本次不修改 `axis_margin`、`eigenvalue_ratio` 或投影峰值分界，避免同时改变候选提取和方向决策两个变量。

## 7. 配置变更

`Config` 新增：

```python
color_support_kernel_size: int = 12
```

以下参数保持不变：

```text
bright_low=(0, 0, 212)
bright_high=(179, 110, 255)
min_component_fill_density=0.15
component_width=(15, 90)
component_height=(12, 90)
min_component_area=80
min_color_score=200
min_color_density=0.10
```

如果后续需要调参，应优先调整 `color_support_kernel_size`，不要先放宽全局亮区阈值。

## 8. 日志设计

每条周期诊断增加：

```text
support_kernel=25
bright_raw_pixels=...
color_support_pixels=...
bright_supported_pixels=...
selection_score=...
```

候选淘汰统计继续保留。默认日志仍然只在以下情况输出：

- 首帧；
- 分类状态变化；
- 每隔 `log_interval` 秒；
- `--debug` 时逐帧输出。

不在 INFO 日志中打印所有候选。需要排查候选竞争时，在 DEBUG 日志中最多输出按 `selection_score` 排序后的前三个候选，包含：

```text
bbox, color, color_score, color_density,
component_area, component_fill, selection_score,
orientation, peak
```

## 9. 验证结果

使用设计方案进行离线模拟，核尺寸按 `12 × scale` 缩放并取奇数：

| 图片 | 当前结果 | 设计方案结果 |
| --- | --- | --- |
| 右箭头 `capture_...000720.jpg` | `unknown` | `right` |
| 左箭头 `capture_...000839.jpg` | `unknown` | `left` |
| 直行 `capture_...000920.jpg` | `straight` | `straight` |

对工作区内之前的 13 张 JPG 样本进行回归，分类结果保持不变；其中原本为 `unknown` 的样本仍为 `unknown`，不会因为本方案被强行分类。

实现完成后的验证结果：三张问题图片和 13 张既有图片共 16 张全部符合预期；
`1280×960` 右箭头图片稳定阶段的单帧检测平均耗时约 27 ms。

## 10. 测试计划

### 10.1 单元测试

1. `color_support_kernel_size` 能随 `scale` 生成正确的奇数核。
2. 颜色支撑掩膜包含红色和绿色区域。
3. 与白色背景粘连的水平箭头能在支撑掩膜中形成合格候选。
4. 远离颜色区域的白色物体不会成为候选。
5. 红灯样本仍输出 `stop`。
6. 多候选情况下，完整度加权排序选择完整箭头。
7. 新增日志字段齐全。

### 10.2 图片回归

至少验证：

- 三张本次问题图片；
- 工作区内全部既有 JPG；
- 红灯、左、右、直行各至少一张；
- 低分辨率 `scale<1`、参考分辨率 `scale=1` 和高分辨率 `scale=2`。

### 10.3 实机验证

在固定相机参数下分别测试：

1. 左、右、直行各持续显示 10 秒；
2. 在当前稳定距离和更远距离各测试一次；
3. 记录 `unknown` 比例、错误方向比例和候选数；
4. 确认检测耗时没有明显破坏实时帧率；
5. 移动白色挡板，确认分类不随背景白色物体发生跳变。

## 11. 验收标准

1. 三张问题图片分类全部正确。
2. 之前图片的已知分类不发生回归。
3. 红灯检测保持正常。
4. 左右箭头不再因为 `width` 超限返回 `unknown`。
5. 右箭头不再因竖向碎片输出 `straight`。
6. 实机稳定画面中，同一方向连续 10 秒不出现错误方向跳变。
7. 默认日志量保持现有水平。

## 12. 实施顺序

1. 扩展 `Config`、`Masks`、`Candidate` 和诊断数据结构。
2. 在 `build_masks()` 中生成颜色支撑掩膜和受约束亮区。
3. 修改候选排序规则。
4. 扩展 INFO/DEBUG 日志。
5. 增加单元测试和三张问题图片回归测试。
6. 完成实机稳定性验证后再考虑是否需要时间序列投票。

## 13. 风险与回退

### 支撑核过小

箭头中心离绿色边缘较远时，亮区可能被截断。表现为 `bright_raw_pixels` 正常，但 `bright_supported_pixels` 明显过少。

处理方式：以小步长增加 `color_support_kernel_size`，例如 `12 → 13`。

### 支撑核过大

附近白色物体可能再次进入支撑区。表现为候选宽度突然增大，或 `accepted_candidates` 显著增加。

处理方式：减小核尺寸，不应通过增大最大候选宽度掩盖问题。

### 颜色掩膜在极端光照下消失

颜色支撑依赖红绿掩膜。如果颜色阈值完全失效，受约束亮区也会消失。后续可增加严格/宽松颜色阈值两阶段策略，但不纳入本次变更，避免一次修改过多变量。

如果新方案发生不可接受的回归，可以通过配置关闭颜色支撑，临时恢复当前全局亮区流程；正式版本默认应启用颜色支撑。
