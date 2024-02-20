"""
Local path planning algorithm for FSAE race car.
author: muzing <muzi2001@foxmail.com>
doc: https://github.com/muziing/PythonAutomatedDriving

原始数据处理：对原始锥桶数据进行整理处理；
边界线拟合：由离散的锥桶位置拟合出连续平滑的车道边界线；
局部路径规划：以外侧车道线向内平移适当距离作为期望路径；
"""

import math
import sys

import numpy as np
from scipy.spatial.distance import cdist, euclidean
from scipy.special import comb


def get_traffic_cone_matrix(raw_data_matrix: np.ndarray, color: int) -> np.ndarray:
    """
    处理原始数据，返回特定类型的锥桶位置矩阵 \n
    :param raw_data_matrix: 由传感器返回的原始数据矩阵
    :param color: 锥桶颜色类型，2-红 11-蓝 13-黄
    :return: 锥桶位置矩阵
    """

    traffic_cone = raw_data_matrix[np.where(raw_data_matrix[:, 1] == color), 2:][0, :]

    if traffic_cone.size < 4:  # 若只有单个锥桶，则直接返回
        return traffic_cone

    sorted_traffic_cone = sort_traffic_cone(traffic_cone, mode=0)  # 使用排序算法排序
    # sorted_traffic_cone = sorted_traffic_cone[0:5, :]  # 出于求解性能之考虑，抛弃第5个点之外的点（最高为4阶贝塞尔曲线）
    clean_traffic_cone = filter_traffic_cone(sorted_traffic_cone)  # 剔除异常的锥桶

    return clean_traffic_cone


def sort_traffic_cone(traffic_cone: np.ndarray, mode: int = 0) -> np.ndarray:
    """
    对锥桶坐标列表进行排序 \n
    :param traffic_cone: 待排序的锥桶位置矩阵
    :param mode: 选用的排序方式
    :return: 排序后的锥桶位置矩阵
    """

    sorted_cones = np.zeros_like(traffic_cone)  # 最终排序好的坐标矩阵

    if mode == 0:
        # 方式一：按x坐标排序
        sorted_cones = traffic_cone[np.lexsort(traffic_cone[:, ::-1].T)]
    elif mode == 1:
        # 方式二：距离第1个锥桶最近的为第2个锥桶，距离第2个锥桶最近的为第3个锥桶……以此类推
        # 创建变量
        sorted_index = list()  # 存储所有已排序锥桶的索引
        unsorted_index = list(range(traffic_cone.shape[0]))  # 还未排序的所有锥桶的索引
        min_distance = sys.maxsize  # 以极大的数初始化最小距离，避免干扰
        index_flag = -1
        counter = 0

        # 找到第1个锥桶
        fist_cone_index = int(np.argmin(traffic_cone[:, 0]))  # x坐标最小的作为首个锥桶
        sorted_index.append(fist_cone_index)
        unsorted_index.remove(fist_cone_index)

        # 距离第1个锥桶最近的为第2个锥桶，距离第2个锥桶最近的为第3个锥桶……以此类推
        while unsorted_index:
            for index in unsorted_index:
                # 计算欧几里德距离
                distance = euclidean(
                    traffic_cone[index, :], traffic_cone[sorted_index[-1], :]
                )
                if distance < min_distance:
                    min_distance = distance
                    index_flag = index
            sorted_index.append(index_flag)
            unsorted_index.remove(index_flag)
            min_distance = sys.maxsize

        # 按排序好的索引顺序拷贝数据，获得最终排序好的坐标矩阵
        for cone in sorted_index:
            sorted_cones[counter] = traffic_cone[cone, :]
            counter += 1

    return sorted_cones


def filter_traffic_cone(traffic_cone: np.ndarray) -> np.ndarray:
    """
    过滤剔除异常的锥桶 \n
    :param traffic_cone: 按序排列的锥桶位置矩阵
    :return: 剔除异常锥桶后的位置矩阵
    """

    delta_y = traffic_cone[1:, 1] - traffic_cone[:-1, 1]
    # 判断弯道方向
    for y in delta_y:
        if abs(y) < 0.25:
            continue
        else:
            direction = math.copysign(1.0, y)
            break
    else:
        direction = 0  # 如果循环结束而没有中途被break，则方向为直行

    # 剔除Δy异常的锥桶
    while direction != 0:
        delta_y = traffic_cone[1:, 1] - traffic_cone[:-1, 1]
        error_index = np.where(delta_y * direction < -0.4)[0]  # 出现异常的锥桶的索引
        if error_index.size > 0:
            traffic_cone = np.delete(
                traffic_cone, error_index[0] + 1, axis=0
            )  # 移除首个异常锥桶
        else:
            break

    return traffic_cone


def local_path_fitting(
    traffic_cone_l: np.ndarray,
    traffic_cone_r: np.ndarray,
    interpolation_num: int = 25,
    translation_dis: float = 1.5,
) -> np.ndarray:
    """
    局部路径规划算法 \n
    :param traffic_cone_l: 左侧锥桶坐标
    :param traffic_cone_r: 右侧锥桶坐标
    :param interpolation_num: 贝塞尔曲线插值点的数量（用于平滑曲线）
    :param translation_dis: 只有单侧锥桶时，向内平移的距离，[m]
    :return: 期望路径
    """

    # 只能看到一侧锥桶时，直接对该侧进行平滑拟合、平移固定距离
    if traffic_cone_l.size == 0:
        bezier_points = get_bezier_curve(traffic_cone_r, interpolation_num)
        translation_points = get_translation(bezier_points, "l", translation_dis)
    elif traffic_cone_r.size == 0:
        bezier_points = get_bezier_curve(traffic_cone_l, interpolation_num)
        translation_points = get_translation(bezier_points, "r", translation_dis)
    else:
        # 能看到两侧锥桶时，以x方向延伸更远的一侧为主，进行平滑拟合、平移距离由计算出的路宽决定
        if traffic_cone_l[-1, 0] > traffic_cone_r[-1, 0]:
            bezier_points = get_bezier_curve(traffic_cone_l, interpolation_num)
            road_width = get_road_width(bezier_points, traffic_cone_r)
            translation_points = get_translation(bezier_points, "r", road_width / 2)
        else:
            bezier_points = get_bezier_curve(traffic_cone_r, interpolation_num)
            road_width = get_road_width(bezier_points, traffic_cone_l)
            translation_points = get_translation(bezier_points, "l", road_width / 2)

    path_points = bezier_points + translation_points

    return path_points


def get_bezier_curve(points: np.ndarray, interpolation_num: int = 25) -> np.ndarray:
    """
    计算贝塞尔曲线 \n
    ref: https://zhuanlan.zhihu.com/p/409585038 \n
    :param points: 控制点
    :param interpolation_num: 控制点间的插值个数
    :return: 贝塞尔曲线
    """

    # 生成对应二项式系数
    n = points.shape[0] - 1
    k = np.arange(0, n + 1, 1)
    b = comb(n, k)

    # 计算权重
    n = points.shape[0] - 1
    sep = 1 / interpolation_num
    t = np.arange(0, 1 + sep, sep)
    t = t.reshape(-1, 1)
    k = np.arange(0, n + 1, 1)
    weight = np.power(t, k) * np.power(1 - t, n - k) * b

    # 计算贝塞尔曲线
    x = np.sum(weight * points[:, 0], axis=1)
    y = np.sum(weight * points[:, 1], axis=1)
    line = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)

    return line


def get_road_width(smooth_curve: np.ndarray, traffic_cone_matrix: np.ndarray) -> float:
    """
    计算道路宽度 \n
    :param smooth_curve: 车道一侧拟合出的平滑曲线坐标
    :param traffic_cone_matrix: 车道另一侧离散的锥桶坐标
    :return: 道路宽度，[m]
    """

    distance_matrix = cdist(smooth_curve, traffic_cone_matrix, "euclidean")
    road_width = distance_matrix.min(initial=None)

    return road_width


def get_translation(
    original_points: np.ndarray, direction: str, distance: float = 1.5
) -> np.ndarray:
    """
    辅助函数，将曲线上的所有点按一定方向平移 \n
    :param original_points: 原始曲线（二维矩阵）
    :param direction: 平移方向，right(r)或left(l)
    :param distance: 平移距离，[m]
    :return: 平移矩阵，需要与原始曲线矩阵相加
    """

    y_dis = distance
    x_dis = -0.5 * y_dis

    if direction == "r" or direction in {"right", "R"}:
        translation = np.tile(np.array([x_dis, -y_dis]), (original_points.shape[0], 1))
    elif direction == "l" or direction in {"left", "L"}:
        translation = np.tile(np.array([x_dis, y_dis]), (original_points.shape[0], 1))
    else:
        translation = np.zeros_like(original_points)

    return translation


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from test_data import test_data  # 加载测试数据

    # 绘图
    fig, axs = plt.subplots(3, 4)
    fig.suptitle("Local Path Planning")
    fig.subplots_adjust(wspace=0.1, hspace=0.5)

    for i in range(len(test_data)):
        red = get_traffic_cone_matrix(np.array(test_data[i]), 2)  # 左侧锥桶位置矩阵
        blue = get_traffic_cone_matrix(np.array(test_data[i]), 11)  # 右侧锥桶位置矩阵
        local_path = local_path_fitting(
            red, blue, translation_dis=1.5
        )  # 规划出的局部路径矩阵

        row = i // 4  # 绘图布局用
        col = i % 4

        # 出于观察方便之考虑，屏幕竖直向上为x正方向
        axs[row, col].plot(red[:, 1] * -1, red[:, 0], "o-", color="r")
        axs[row, col].plot(blue[:, 1] * -1, blue[:, 0], "o-", color="b")
        axs[row, col].plot(
            local_path[:, 1] * -1, local_path[:, 0], color="k", linestyle="--"
        )
        axs[row, col].set_aspect("equal")
        axs[row, col].set_title(f"{i}")

    plt.show()
