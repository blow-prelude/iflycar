# YOLO 定位与传统视觉交通灯分类集成设计

## 目标

修改 `src/traffic_light/scripts/judge_light_ros_correct.py`，保留现有 YOLO/RKNN 推理作为交通灯候选框定位器，并用 `traditional_light_cv.py` 已验证的传统 OpenCV 算法完成最终交通指令分类。

最终分类支持 `stop`、`straight`、`left`、`right` 和 `unknown`。只有前四种有效结果可以发布；`unknown` 必须跳过当前帧，不得回退到 YOLO 类别。

## 约束

- 传统视觉实现直接写入 `judge_light_ros_correct.py`。
- 不导入或调用 `traditional_light_cv.py`、`traditional_light_cv_ros.py`。
- YOLO/RKNN 的预处理、推理线程、后处理和候选框输出保持不变。
- YOLO 类别只随现有后处理产生，不参与最终方向决策。
- 保持现有 ROS 参数、相机订阅话题、方向发布话题、发布节流及退出行为。
- 不修改 `traditional_light_cv.py` 和 `traditional_light_cv_ros.py` 的既有行为。

## 方案选择

采用“YOLO 框限定搜索区域，在原始 BGR 图上运行传统视觉”的方案。

未采用的方案：

- 裁剪 YOLO 框并缩放到固定尺寸：缩放和插值会改变 LED 亮核连通性、面积阈值和箭头宽高比例。
- YOLO 仅触发传统视觉扫描整幅图：没有利用 YOLO 定位结果隔离背景干扰。

选定方案既保留 YOLO 对位置变化的适应能力，也保留传统视觉在原始像素上的颜色和形状判别效果。

## 文件范围

修改：

- `src/traffic_light/scripts/judge_light_ros_correct.py`

新增：

- `src/traffic_light/test/test_judge_light_traditional_cv.py`

测试文件只验证 `judge_light_ros_correct.py` 内直接定义的纯 OpenCV 辅助逻辑。测试可为 ROS、RKNN 和消息模块提供最小桩对象，但不得从另外两个传统视觉脚本导入实现。

## 组件设计

### YOLO/RKNN 定位层

沿用现有三模型推理线程、`post_process`、置信度阈值和最高分候选选择。若没有候选框、框数组无效或最高分低于 `OBJ_THRESH`，当前帧不进入传统分类。

最高分候选的 letterbox 坐标通过 `COCO_test_helper.get_real_box` 映射回原始相机图像。映射结果按图像边界裁剪；宽或高不足的无效框返回 `unknown`。

YOLO 的 `best_class` 不再决定发布值，也不再参与左右方向复核。最终分类接口不接收 YOLO 类别，从结构上避免回退或误用。

### 传统视觉分类层

在 `judge_light_ros_correct.py` 内直接定义传统视觉配置、检测结果和纯函数。核心入口接收原始 BGR `uint8` 图像及原图坐标系中的 YOLO `xyxy` 区域，返回包含最终标签和亮核框的检测结果。

算法与现有传统视觉实现保持一致：

1. 根据整幅原图相对 640×480 的比例计算长度和面积阈值。
2. 将 YOLO 区域按配置值扩边并裁剪到图像边界。
3. 在原始 BGR 图上生成以下 HSV 掩膜：
   - 绿色：`H=35..100, S>=80, V>=100`
   - 红色一：`H=0..12, S>=80, V>=100`
   - 红色二：`H=165..179, S>=80, V>=100`
   - 亮核：`S<=110, V>=220`
4. 只允许扩边后的 YOLO 区域产生亮核候选；颜色掩膜保留候选邻域所需的原始像素。
5. 对亮核掩膜执行 3×3 椭圆核闭运算和连通域分析。
6. 按缩放后的宽、高、面积、邻域颜色得分和颜色密度过滤候选，并选择颜色得分最高者。
7. 红色得分更高时输出 `stop`。
8. 绿色得分更高时，对亮核连通域执行 PCA：主轴竖直输出 `straight`；主轴水平时继续进行八段投影。
9. 八段投影使用 `numpy.array_split` 等分宽度，并比较每段前景密度，避免非八倍数宽度造成偏置。唯一峰值位于 0～3 段时输出 `left`，位于 4～7 段时输出 `right`；峰值并列时输出 `unknown`。

候选不存在、红绿得分相等、颜色不足、PCA 无效或投影不唯一时均返回 `unknown`。

### ROS 控制层

传统分类返回有效标签后，继续使用原有计数器节流：每累计 10 次有效识别发布一次。`unknown` 不增加有效识别发布计数，也不发布消息。

发布到 `~direction_topic`，默认值仍为 `/vision_line_direction`。发布 `stop` 后节点继续识别；发布 `left`、`right` 或 `straight` 后等待 1 秒并退出主循环，与当前控制流程一致。

### 调试显示与日志

显示画面以原始 BGR 帧为底图，绘制传统视觉选中的亮核框和最终标签，并保留 FPS。不得把 YOLO 类别作为最终标签绘制，避免画面与发布值冲突。

日志分别保留定位和分类诊断：

- YOLO：worker、最高候选分数、推理耗时。
- 传统视觉：最终标签、亮核框、颜色、颜色得分、连通域面积、PCA 方向和投影峰值。
- `unknown`：以节流方式记录原因或结果，但不发布。

## 数据流

```text
ROS Image (BGR)
  -> letterbox RGB
  -> YOLO/RKNN inference
  -> boxes + scores
  -> highest-score box
  -> map box back to original BGR image
  -> clamp and pad search region
  -> HSV green/red/bright masks
  -> bright connected components
  -> size and color-neighborhood filtering
  -> best bright-core candidate
     -> red   -> stop
     -> green -> PCA
                  -> vertical   -> straight
                  -> horizontal -> normalized 8-band projection
                                      -> left/right/unknown
  -> unknown: skip frame
  -> valid label: existing publish throttle and exit policy
```

## 错误处理

- ROS 图像转换失败：记录错误并等待下一帧，沿用当前行为。
- YOLO 无结果、低置信度或坐标无效：跳过当前帧。
- 传统视觉无法可靠分类：返回 `unknown` 并跳过，不使用 YOLO 类别兜底。
- OpenCV 单帧处理异常：记录错误并继续主循环，不终止节点。
- 节点关闭或发生不可恢复异常：设置停止事件、等待推理线程并释放三个 RKNN 模型，沿用当前清理逻辑。

## 测试设计

新增标准库 `unittest` 测试，至少覆盖：

1. 四张现有样例图在包含目标的模拟 YOLO 区域内分别输出 `right`、`straight`、`left`、`stop`。
2. 全黑图、有框但无合格亮核时返回 `unknown`。
3. 空框、倒置框、完全越界框返回 `unknown`；部分越界框被安全裁剪。
4. 合成的左、右、直行箭头掩膜得到预期方向。
5. 八段宽度不等时按密度而不是像素总数选峰，防止方向偏置回归。
6. 并列峰值返回 `unknown`。
7. 最高分 YOLO 框被选中，且传统分类入口不读取 YOLO 类别。
8. 源码中不存在对 `traditional_light_cv` 或 `traditional_light_cv_ros` 的导入。

本地验证包括 Python 语法编译、传统视觉单元测试和既有 `traditional_light_cv`/`traditional_light_cv_ros` 回归测试。ROS、相机与 RKNN 的完整联调需在目标设备执行：确认 `/vision_line_direction` 只发布传统视觉的有效结果，`unknown` 不发布，且非 `stop` 发布后节点按原逻辑退出。

## 验收标准

- `judge_light_ros_correct.py` 仍由 `track_test.sh` 原命令启动，无需改启动脚本。
- YOLO/RKNN 只定位，最终方向完全由文件内直接加入的传统视觉代码决定。
- 四张样例图回归测试全部通过。
- `unknown` 跳过且绝不回退 YOLO 类别。
- ROS 话题名称、发布节流、`stop` 持续运行和非 `stop` 退出行为保持兼容。
- `traditional_light_cv.py` 与 `traditional_light_cv_ros.py` 无需作为运行时依赖。
