"""
一些补充numpy功能的自实现函数
"""

import math

import numpy as np


def normalize_angle(angle: float) -> float:
    """
    将角度转换到 [-pi, pi] 范围中 \n
    :param angle: 弧度制角度
    :return: [-pi, pi] 范围内的弧度制角度
    """

    return np.arctan2(np.sin(angle), np.cos(angle))


def find_array_nearest(array_1d, value, mode: int = 1, *, sorter=None) -> int:
    """
    在一维数组中搜索与给定值最接近的值的索引 \n
    ref: https://stackoverflow.com/questions/2566412
    :param array_1d: 一维数组
    :param value: 待搜索的值
    :param mode: 默认模式（mode!=0）可靠性高；快速模式（mode==0）要求数组有序或传入排序器
    :param sorter: 数组排序器（仅在对非有序数组使用快速模式时需要）
    :return: index
    """

    array = np.array(array_1d)
    if mode:
        # 普通模式，速度慢，但数组无需排序
        index = np.argmin(abs(array - value))
    else:
        # 快速模式，速度约为普通模式的3倍，要求数组有序，某些情况下可能会出错
        index = np.searchsorted(array, value, side="left", sorter=sorter)
        if index > 0 and (
            index == len(array)
            or math.fabs(value - array[index - 1]) < math.fabs(value - array[index])
        ):
            index -= 1

    return int(index)
