#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动标定透视变换参数工具
从斜视图变换到鸟瞰图（Bird's Eye View）

使用方法：
1. 运行脚本
2. 在图像上依次点击4个点：左上、右上、右下、左下
3. 查看透视变换结果
4. 控制台会打印变换矩阵
"""

import glob
import os

import cv2
import numpy as np


class PerspectiveCalibration:
    """透视变换手动标定工具"""

    def __init__(self, image_path):
        """
        初始化标定工具

        Args:
            image_path: 图片路径
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 如果图片太大，按比例缩小（与 vision_line_test.py 保持一致）
        if self.image.shape[0] >= 240 or self.image.shape[1] >= 320:
            # 将图片按比例缩小，使宽和高都不超过320和240
            h, w = self.image.shape[:2]
            scale = min(240 / h, 320 / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            self.image = cv2.resize(
                self.image, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            print(f"图片已从 {w}x{h} 缩放到 {new_w}x{new_h}")

        self.height, self.width = self.image.shape[:2]
        self.points = []  # 存储用户点击的4个点
        self.display_image = self.image.copy()
        self.window_name = "选择4个点: 左上 -> 右上 -> 右下 -> 左下"

        self.dst_rect = (0.2, 0.8, 0.2, 0.75)  # 目标矩形归一化坐标,分别为 左右上下

    def mouse_callback(self, event, x, y, flags, param):  # noqa: ARG002
        """
        鼠标回调函数，处理点击事件

        Args:
            event: OpenCV鼠标事件
            x, y: 鼠标坐标
            flags: 事件标志
            param: 附加参数
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                # 添加点
                self.points.append((x, y))

                # 打印点的信息
                point_names = ["左上", "右上", "右下", "左下"]
                print(
                    f"点 {len(self.points)} ({point_names[len(self.points) - 1]}): ({x}, {y})"
                )

                # 在图像上标记点
                cv2.circle(self.display_image, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(
                    self.display_image,
                    f"{len(self.points)}",
                    (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                # 如果有多个点，画线连接
                if len(self.points) > 1:
                    prev_point = self.points[-2]
                    cv2.line(self.display_image, prev_point, (x, y), (255, 0, 0), 2)

                # 显示更新后的图像
                cv2.imshow(self.window_name, self.display_image)

                # 如果已经选择了4个点，自动继续
                if len(self.points) == 4:
                    print("\n4个点已选择完成！")
                    cv2.waitKey(500)  # 短暂延迟让用户看到最后一个点
                    cv2.destroyWindow(self.window_name)

    def select_points(self):
        """交互式选择4个点"""
        print("\n" + "=" * 50)
        print("透视变换标定工具")
        print("=" * 50)
        print(f"\n图片尺寸: {self.width} x {self.height}")
        print("\n请在图像上依次点击4个点：")
        print("1. 左上点 (Top-Left)")
        print("2. 右上点 (Top-Right)")
        print("3. 右下点 (Bottom-Right)")
        print("4. 左下点 (Bottom-Left)")
        print("\n点击顺序很重要！请按顺序点击。")
        print("=" * 50 + "\n")

        # 创建窗口并设置鼠标回调
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # 显示图像
        cv2.imshow(self.window_name, self.display_image)
        print("等待点击... (按ESC退出)")

        # 等待用户完成选择或按ESC退出
        while len(self.points) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键
                print("\n用户取消操作")
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()

        # 打印所有点的总结
        print("\n" + "=" * 50)
        print("已选择的源点 (Source Points):")
        print("=" * 50)
        point_names = ["左上 (TL)", "右上 (TR)", "右下 (BR)", "左下 (BL)"]
        for name, point in zip(point_names, self.points):
            print(f"  {name}: {point}")
        print("=" * 50 + "\n")

        return self.points

    def create_perspective_transform(self, src_points):
        """
        创建透视变换并应用

        Args:
            src_points: 源点列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]

        Returns:
            transformation_matrix: 3x3透视变换矩阵
            warped_image: 变换后的图像
        """
        # 定义目标点（鸟瞰图的矩形区域）
        # 可以自定义输出尺寸，这里使用原图尺寸
        output_width = self.width
        output_height = self.height

        # 目标点：矩形，顺序为 左上、右上、右下、左下
        dst_points = np.array(
            [
                [
                    output_width * self.dst_rect[0],
                    output_height * self.dst_rect[2],
                ],  # 左上
                [
                    output_width * self.dst_rect[1],
                    output_height * self.dst_rect[2],
                ],  # 右上
                [
                    output_width * self.dst_rect[1],
                    output_height * self.dst_rect[3],
                ],  # 右下
                [
                    output_width * self.dst_rect[0],
                    output_height * self.dst_rect[3],
                ],  # 左下
            ],
            dtype=np.float32,
        )

        # 源点转换为numpy数组
        src_points_array = np.array(src_points, dtype=np.float32)

        # 生成透视变换矩阵
        transformation_matrix = cv2.getPerspectiveTransform(
            src_points_array,
            dst_points,  # type: ignore[arg-type]
        )

        # 应用透视变换
        assert self.image is not None  # 已在__init__中检查
        warped_image = cv2.warpPerspective(
            self.image,
            transformation_matrix,
            (output_width, output_height),  # type: ignore[arg-type]
        )

        return transformation_matrix, warped_image

    def display_results(self, warped_image):
        """
        显示标定结果

        Args:
            warped_image: 透视变换后的图像
        """
        # 创建包含原图和变换图的并排显示
        # 在原图上画出选中的区域
        assert self.image is not None  # 已在__init__中检查
        result_image = self.image.copy()

        # 画出四边形区域（半透明）
        overlay = result_image.copy()
        pts = np.array(self.points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, result_image, 0.7, 0, result_image)

        # 画出四边形边框
        cv2.polylines(result_image, [pts], True, (0, 255, 0), 2)

        # 标注点
        for i, point in enumerate(self.points):
            cv2.circle(result_image, point, 5, (0, 0, 255), -1)
            cv2.putText(
                result_image,
                str(i + 1),
                (point[0] - 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # 调整图像大小以便并排显示
        h, w = result_image.shape[:2]
        result_resized = cv2.resize(result_image, (w // 2, h // 2))
        warped_resized = cv2.resize(warped_image, (w // 2, h // 2))

        # 水平拼接
        combined = np.hstack((result_resized, warped_resized))

        # 添加标题
        combined = cv2.copyMakeBorder(
            combined, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        cv2.putText(
            combined,
            "Original with Selected Points",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            combined,
            "Bird's Eye View (Warped)",
            (w // 2 + 10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

        # 显示结果
        cv2.imshow("Perspective Transform Result", combined)
        print("\n按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(self):
        """运行标定流程"""
        try:
            # 步骤1: 选择点
            src_points = self.select_points()

            if src_points is None or len(src_points) != 4:
                print("标定取消或点数不足")
                return

            # 步骤2: 创建透视变换
            print("生成透视变换矩阵...")
            transformation_matrix, warped_image = self.create_perspective_transform(
                src_points
            )

            # 步骤3: 打印变换矩阵
            print("\n" + "=" * 50)
            print("透视变换矩阵 (Perspective Transform Matrix):")
            print("=" * 50)
            print(transformation_matrix)
            print("=" * 50)

            # 打印可用的Python代码格式
            print("\n复制以下代码到你的项目中使用此变换矩阵：\n")
            print("transformation_matrix = np.array([")
            for row in transformation_matrix:
                print(f"    [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}],")
            print("])")

            # 步骤4: 显示结果
            self.display_results(warped_image)

            print("\n标定完成！")

        except Exception as e:
            print(f"\n错误: {e}")
            import traceback

            traceback.print_exc()


def find_test_image(pictures_dir):
    """
    查找测试图片

    Args:
        pictures_dir: 图片目录路径

    Returns:
        图片路径或None
    """
    # 常见的图片格式
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    for ext in extensions:
        pattern = os.path.join(pictures_dir, ext)
        files = glob.glob(pattern)
        if files:
            return files[0]  # 返回第一个找到的图片

    return None


def main():
    """主函数"""
    # 获取图片目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pictures_dir = os.path.join(os.path.dirname(script_dir), "pictures")

    # 查找测试图片
    image_path = r"D:\programs\ucar_ws\src\vision_line\pictures\test5.png"

    if image_path is None:
        print(f"错误: 在 {pictures_dir} 目录下没有找到图片")
        print("请将测试图片放到该目录下，支持格式: jpg, jpeg, png, bmp")
        return

    print(f"使用图片: {image_path}")

    # 创建标定工具并运行
    calibration = PerspectiveCalibration(image_path)
    calibration.run()


if __name__ == "__main__":
    main()
