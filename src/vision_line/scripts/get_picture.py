#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片旋转工具
给定图片路径、旋转方向和角度，生成旋转后的图像
"""

import os

import cv2


def rotate_image(image_path, direction="clockwise", angle=90, output_path=None):
    """
    旋转图片并保存

    Args:
        image_path: 输入图片路径
        direction: 旋转方向 ('clockwise' 顺时针, 'counterclockwise' 逆时针)
        angle: 旋转角度 (0-360)
        output_path: 输出路径 (如果为None，则在原路径添加后缀)

    Returns:
        output_path: 输出图片的路径
    """
    # 验证输入
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    if angle < 0 or angle > 360:
        raise ValueError("角度必须在0-360之间")

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    # 根据方向确定旋转角度
    # OpenCV的rotate函数正值表示逆时针旋转
    if direction == "clockwise":
        rotate_angle = -angle
    elif direction == "counterclockwise":
        rotate_angle = angle
    else:
        raise ValueError("方向必须是 'clockwise' 或 'counterclockwise'")

    # 获取旋转矩阵
    # 第一个参数是旋转中心，第二个参数是旋转角度，第三个参数是缩放因子
    M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)

    # 计算旋转后的新图像大小，确保完整显示旋转后的图像
    # 使用边界框计算新的尺寸
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    # 调整旋转矩阵以考虑平移
    M[0, 2] += (new_width - width) / 2
    M[1, 2] += (new_height - height) / 2

    # 执行旋转
    rotated_img = cv2.warpAffine(
        img,
        M,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # 生成输出路径
    if output_path is None:
        base_name = os.path.splitext(image_path)[0]
        ext = os.path.splitext(image_path)[1]
        direction_suffix = "cw" if direction == "clockwise" else "ccw"
        output_path = f"{base_name}_rotated_{direction_suffix}_{angle}{ext}"

    # 保存图片
    cv2.imwrite(output_path, rotated_img)
    print(f"旋转后的图片已保存至: {output_path}")

    return output_path


if __name__ == "__main__":
    # ========== 参数设置 ==========
    # 输入图片路径
    image_path = r"src\vision_line\scripts\raw1.png"

    # 旋转方向: 'clockwise' 顺时针, 'counterclockwise' 逆时针
    direction = "clockwise"

    # 旋转角度 (0-360)
    angle = 90

    # 输出路径 (留空则在原路径添加后缀)
    output_path = None  # 或指定路径: r'D:\path\to\output.jpg'

    # ========== 执行旋转 ==========
    rotate_image(image_path, direction, angle, output_path)
