# 传统视觉交通灯识别 OpenCV 示例设计

## 目标

为 `src/traffic_light/pictures` 中的四张 640×480 图像提供一个不依赖 YOLO、RKNN 或训练数据的 Python/OpenCV 示例，识别并标注以下四类交通灯状态：

- `02051.jpg`：`right`
- `02052.jpg`：`straight`
- `003_0030.jpg`：`left`
- `004_0001.jpg`：`stop`

示例入口从代码内集中定义的相对路径读取四张图片，打印识别结果，并保存只包含交通灯亮核候选框和最终类别的标注图。代码使用用户指定的 `D:\Anaconda\envs\opencv39\python.exe` 运行。

## 范围

本次实现包含：

- 纯 OpenCV/NumPy 的候选定位、颜色分类和箭头方向分类。
- 一个不接收命令行参数、可直接运行的固定图片示例入口。
- 针对四张样例图的自动化回归测试。
- 对空白图、无法读取的文件和无法可靠分类的候选给出明确结果。

本次不包含：

- ROS 订阅、发布或启动文件集成。
- C++ 实现及其链接配置。
- 摄像头实时采集。
- 命令行参数解析。
- 对任意道路场景、任意交通灯外观的泛化保证。
- 修改或移除现有 YOLO/RKNN 代码。

## 方案选择

采用“亮核候选 + HSV 判色 + PCA/投影判方向”。交通灯箭头和停止灯的主体具有低饱和、高亮度的发光核心，核心周围存在明显的绿色或红色光晕。先找亮核，再检查邻域颜色，可以避免直接从整幅绿色或红色掩膜中选择连通域时被地面反光吸引。

未采用的方案：

- 模板匹配：对固定灯牌有效，但尺度、曝光和透视变化会明显降低匹配分数，并需要维护模板图。
- 纯轮廓几何规则：LED 点阵间隙、过曝和反光容易使轮廓断裂或粘连，单独使用稳定性不足。

## 文件结构与接口

新增文件：

- `src/traffic_light/scripts/traditional_light_cv.py`
- `src/traffic_light/test/test_traditional_light_cv.py`

核心接口：

```python
detect_traffic_light(image: np.ndarray) -> Detection
```

`image` 必须是非空的 BGR `uint8` 图像。`Detection` 至少包含：

- `label`：`stop`、`straight`、`left`、`right` 或 `unknown`。
- `bbox`：候选亮核的 `(x, y, width, height)`；无候选时为 `None`。
- `color`：`red`、`green` 或 `unknown`。
- `color_score`：候选邻域内胜出颜色的像素数。
- `component_area`：亮核连通域面积。
- `orientation`：`vertical`、`horizontal` 或 `unknown`。
- `projection_peak`：水平方向八段投影的峰值段编号；不适用时为 `None`。

辅助接口保持单一职责：

- `build_masks`：从 HSV 图生成 ROI、绿色、红色和亮核掩膜。
- `find_candidate`：过滤亮核连通域并按邻域颜色得分选择候选。
- `classify_green_shape`：使用 PCA 和列投影识别绿色箭头。
- `draw_detection`：在副本上只绘制亮核候选框和最终类别。

模块集中定义以下项目相对路径常量，不使用机器相关的绝对路径：

- `SAMPLE_IMAGE_PATHS`：四张样例图的路径元组。
- `OUTPUT_DIR`：工作区 `build/traditional_light_cv_results` 目录。

`process_images(image_paths, output_dir)` 负责批量读取、检测和保存，便于测试时替换输出目录；`main()` 不接收参数，直接以 `SAMPLE_IMAGE_PATHS` 和 `OUTPUT_DIR` 调用该函数。文件读取与单帧检测保持分离，因此后续相机回调可以直接调用 `detect_traffic_light(frame)`，无需经过磁盘或示例入口。

## 检测流程

### 1. 输入与尺度归一化

以 640×480 为参数参考分辨率。对于其他分辨率，使用：

```text
scale = min(width / 640, height / 480)
```

将面积阈值按 `scale²` 缩放，将长度和扩展边距按 `scale` 缩放。算法不主动缩放输入图，以避免插值改变小型 LED 灯的颜色和连通性。

### 2. 搜索区域

根据当前固定相机和四张样例图，搜索区域只以 `[0, 1]` 范围的归一化坐标保存：

```text
x ∈ [0.25 × width, 0.78 × width)
y ∈ [0.28 × height, 0.78 × height)
```

每次调用 `detect_traffic_light(frame)` 时，根据当前帧宽高将归一化 ROI 换算为整数像素边界，再把区域之外的掩膜清零。该限制排除大部分天花板灯和画面边缘杂物，同时覆盖四张图中的交通灯。ROI 参数集中定义，不能以 640×480 的绝对像素坐标保存，以便后续直接处理分辨率不同的相机帧。

### 3. HSV 掩膜

OpenCV HSV 通道范围为 `H: 0..179`、`S/V: 0..255`。初始阈值为：

```text
green: H=35..100, S>=80, V>=100
red-1: H=0..12,   S>=80, V>=100
red-2: H=165..179,S>=80, V>=100
bright core: S<=110, V>=220
```

红色掩膜由两个色相区间取并集。亮核不限制色相，以保留绿色箭头内部接近白色的 LED 和红灯过曝形成的白色矩形。

亮核掩膜使用 3×3 椭圆核做一次闭运算，连接相邻 LED 点；不对整幅颜色掩膜做大尺度膨胀，避免灯牌与地面反光粘连。

### 4. 候选定位

对亮核掩膜运行 `connectedComponentsWithStats`。以参考分辨率计，候选需要满足：

```text
width: 15..90 px
height: 12..90 px
area: >= 80 px²
```

每个候选框向外扩展 `10 × scale` 像素，分别统计扩展区域内的绿色和红色像素数。候选得分为两者的较大值，颜色为得分较高的一方。

候选还需同时满足：

- `color_score >= 200 × scale²`
- `color_score / expanded_area >= 0.10`

在合格候选中选择 `color_score` 最大者。此设计利用“白色发光核心附近必须存在足够的红色或绿色光晕”，用于排除白色墙板、顶灯和其他高亮区域。若没有合格候选，返回 `unknown`。

### 5. 颜色分类

若候选邻域的红色得分大于绿色得分，直接输出 `stop`。停止灯在 `004_0001.jpg` 中中心严重过曝，因此不能只依赖红色主体面积；邻域红色光晕负责完成判色。

若绿色得分更高，进入箭头形状分类。

### 6. 箭头形状分类

使用候选亮核连通域的所有像素坐标计算二维协方差矩阵，并取得最大特征值对应的主轴向量 `(vx, vy)`：

- `abs(vy) >= abs(vx)`：主轴竖直，输出 `straight`。
- `abs(vx) > abs(vy)`：主轴水平，继续判断左右方向。

对水平箭头的候选掩膜按宽度等分成八段，统计每段前景像素数。投影最大值只有一个时，将其下标作为峰值段；峰值位于第 0～3 段时输出 `left`，位于第 4～7 段时输出 `right`。若多个段并列最大，则返回 `unknown`。

四张图的只读像素分析得到：

- `02052.jpg` 的亮核主轴近似竖直。
- `003_0030.jpg` 的列投影峰值位于第 2 段。
- `02051.jpg` 的列投影峰值位于第 5 段。

若候选点数不足、协方差矩阵无效或投影无法形成唯一峰值，返回 `unknown`，不猜测交通指令。

## 数据流

```text
BGR image
  -> normalized ROI
  -> HSV green/red/bright masks
  -> bright-mask closing
  -> connected components
  -> size and color-neighborhood filtering
  -> best candidate
     -> red   -> stop
     -> green -> PCA
                  -> vertical   -> straight
                  -> horizontal -> 8-band projection -> left/right
  -> Detection + optional annotated image
```

## 调试输出

终端每张图打印一行，包括文件名、类别、候选框、颜色得分、亮核面积、主轴方向和投影峰值。ROI 边界不打印。

标注图只包含：

- 候选亮核外接框。
- 最终类别。

标注图不绘制 ROI、颜色得分、面积、主轴方向或投影峰值。保存时沿用源文件名并添加 `_traditional` 后缀，写入代码常量 `OUTPUT_DIR` 指定的目录，不覆盖源图片。

## 错误处理

- 固定图像路径不存在或 `cv2.imread` 失败：终端输出明确错误，并在全部图片处理结束后返回非零状态。
- `detect_traffic_light` 收到空图、非三通道图或非 `uint8` 图：抛出 `ValueError`。
- 没有合格候选或形状不可靠：返回 `Detection(label="unknown")`。
- 输出目录无法创建或图片无法写入：输出具体路径并以非零状态退出。
- 多图处理中任意一张失败时继续处理其余图片，`process_images` 的返回值表示是否存在失败项。

## 测试与验收

测试使用 `D:\Anaconda\envs\opencv39\python.exe` 运行，并采用 Python 标准库 `unittest`，避免新增测试框架依赖。

自动测试至少覆盖：

1. 四张样例图分别得到 `right`、`straight`、`left` 和 `stop`。
2. 四个有效结果均有非空候选框，且候选框位于配置的 ROI 内。
3. 全黑 640×480 图返回 `unknown`，不产生误检。
4. 空数组、灰度图和错误数据类型被拒绝。
5. 标注函数不修改传入的原始图像。
6. `process_images` 可处理代码中固定的四张输入图片，并能在测试指定的临时目录生成四张非空标注图。
7. 标注图只新增候选框和最终类别，不绘制 ROI 边界或其他调试指标。
8. `main()` 不解析命令行参数，直接使用 `SAMPLE_IMAGE_PATHS` 和 `OUTPUT_DIR`。

验收命令将在实施计划中给出；最低验收条件是自动测试全部通过，并人工检查四张标注图的框均落在实际交通灯上。

## 已知限制与调参边界

该方案针对当前室内固定相机、固定 LED 灯牌和相近曝光设计。以下变化可能需要调参：

- 交通灯移出相对 ROI。
- 灯牌在图中的尺寸超出候选范围。
- 相机自动曝光使亮核不再满足 `V>=220`。
- 环境中出现与灯牌尺寸相近、同时具有白色核心和彩色光晕的物体。
- 箭头旋转角度过大，导致 PCA 主轴不再接近水平或竖直。

所有阈值集中在一个配置数据结构中，不散落为魔法数字。后续若接入实时相机，应先收集包含距离、曝光和遮挡变化的图片，再调整 ROI、HSV 阈值和候选尺寸；不在本示例中提前增加自适应算法。
