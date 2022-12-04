"""
坐标系变换算法
"""

import math

import numpy as np


def trans_2d_coordinate(
    raw_matrix: np.ndarray,
    rotation: float,
    translation_x: np.ndarray,
    translation_y: np.ndarray,
) -> np.ndarray:
    """
    二维坐标系变换，平移+旋转 \n
    :param raw_matrix: 原始坐标矩阵
    :param rotation: 旋转角
    :param translation_x: x方向平移
    :param translation_y: y方向平移
    :return: 变换后的坐标
    """

    rotation_matrix = np.array(
        [
            [math.cos(rotation), -math.sin(rotation)],
            [math.sin(rotation), math.cos(rotation)],
        ]
    )
    translation_matrix = np.array([translation_x, translation_y])

    return np.dot(raw_matrix, rotation_matrix) + translation_matrix
