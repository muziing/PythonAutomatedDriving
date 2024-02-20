"""
离散曲线相关函数
"""

import math
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from numpy_additional_funcs import normalize_angle


def find_match_point(
    xy_coordinate: tuple[float, float],
    reference_line_nodes: npt.NDArray[np.float_],
    *,
    start_index: int = 0,
) -> Tuple[int, float]:
    """
    曲线由若干离散点组成的点集表达，遍历曲线外某一点到该点集中每一点的距离，取距离最小的为匹配点，返回该点在点集中的索引和最小距离
    :param xy_coordinate: (x, y) 点的坐标
    :param reference_line_nodes: [[x0, y0], [x1, y1], ...] 构成参考线曲线的点集
    :param  start_index: 从点集中的该索引位置开始匹配
    :return: (match_point_index, min_distance) 匹配点在曲线点集中的索引、到匹配点的距离
    """

    x, y = xy_coordinate
    reference_line_length = reference_line_nodes.shape[0]
    increase_count = (
        0  # 用 increase_count 记录 distance 连续增大的次数，避免多个局部最小值的干扰
    )
    min_distance_square = float("inf")
    match_point_index = (
        start_index  # 若最后仍没有找到匹配索引，则说明起始索引已经是最佳匹配，直接返回
    )

    if start_index == 0:
        # 首次运行情况
        increase_count_limit = reference_line_length // 3
        direction_flag = 1  # 正向遍历
    elif start_index < 0:
        raise ValueError("index < 0")
    else:
        # 非首次运行情况
        increase_count_limit = 5

        # 上个周期匹配点坐标
        pre_match_point = [
            reference_line_nodes[start_index, 0],
            reference_line_nodes[start_index, 1],
        ]
        # 以上个周期匹配点、上个周期匹配点的前一个点之间的连线的方向，近似表示切向
        d_x = pre_match_point[0] - reference_line_nodes[start_index - 1, 0]
        d_y = pre_match_point[1] - reference_line_nodes[start_index - 1, 1]
        pre_match_point_theta = np.arctan2(d_y, d_x)

        # 上个匹配点在曲线上的切向向量
        pre_match_point_direction = np.array(
            [
                normalize_angle(np.cos(pre_match_point_theta)),
                normalize_angle(np.sin(pre_match_point_theta)),
            ]
        )

        # 计算上个匹配点指向当前 (x, y) 的向量
        pre_match_to_xy_v = np.array([x - pre_match_point[0], y - pre_match_point[1]])

        # 计算 pre_match_to_xy_v 在 pre_match_point_direction 上的投影，用于判断遍历方向
        direction_flag = np.dot(
            pre_match_to_xy_v, pre_match_point_direction
        )  # 大于零正反向遍历，反之，反方向遍历

    if direction_flag > 0:  # 正向遍历
        search_range = (start_index, reference_line_length, 1)
    else:  # 反向遍历
        search_range = (start_index, -1, -1)

    # 确定匹配点
    for i in range(*search_range):
        reference_line_node_x = reference_line_nodes[i][0]
        reference_line_node_y = reference_line_nodes[i][1]
        # 计算 (x, y) 与 (reference_line_node_x, reference_line_node_y) 之间的距离
        distance_square = (reference_line_node_x - x) ** 2 + (
            reference_line_node_y - y
        ) ** 2
        if distance_square < min_distance_square:
            min_distance_square = distance_square  # 保留最小值
            match_point_index = i
            increase_count = 0
        else:
            increase_count += 1
            if increase_count >= increase_count_limit:
                break

    return match_point_index, math.sqrt(min_distance_square)


def get_projection_point(
    xy_coordinate: tuple[float, float],
    reference_line_nodes: npt.NDArray[np.float_],
    match_point_index: Optional[int] = None,
) -> tuple[float, float, float, float]:
    """
    TODO 此函数待验证
    获取某点在一条由离散点表示的参考线上的投影点信息 \n
    ref: https://www.bilibili.com/video/BV1EM4y137Jv
    :param xy_coordinate: (x, y) 笛卡尔坐标系下点的坐标
    :param reference_line_nodes: [[x0, y0, heading0, kappa0], ...] 参考线上的若干离散点
    :param match_point_index: [可选参数] 匹配点在参考线点集中的索引
    :return: 匹配点的 (x, y, theta, kappa)
    """

    x, y = xy_coordinate

    # d_v 是匹配点（x_match, y_match）指向待投影点（x,y）的向量（x-x_match, y-y_match）
    # tau 是匹配点的单位切向量(cos(theta_match), sin(theta_match))'
    # (x_r, y_r)' 约等于 (x_match, y_match)' + (d_v . tau) * tau
    # kappa_r 约等于 kappa_match，投影点曲率
    # theta_r 约等于 theta_match + k_m * (d_v . tau)，投影点切线与坐标轴夹角

    if match_point_index is None:
        match_point_index, _ = find_match_point(
            xy_coordinate,
            reference_line_nodes[:, 0:2],
            start_index=0,
        )

    # 通过匹配点确定投影点
    x_match, y_match, theta_match, kappa_match = reference_line_nodes[match_point_index]
    d_v = np.array([x - x_match, y - y_match])  # 匹配点指向待投影点的向量
    tau = np.array([np.cos(theta_match), np.sin(theta_match)])  # 匹配点的单位切向量
    ds = np.dot(d_v, tau)
    r_m_v = np.array([x_match, y_match])

    # 计算投影点的位置信息
    x_r, y_r = r_m_v + ds * tau  # 计算投影点坐标
    theta_r = normalize_angle(
        theta_match + kappa_match * ds
    )  # 计算投影点在参考线上切线与 x 轴的夹角
    kappa_r = kappa_match  # 投影点在参考线处的曲率

    return x_r, y_r, theta_r, kappa_r


def calculate_heading_kappa(path_points: npt.NDArray[np.float_]):
    """
    计算曲线上每个点的切向角 theta（与直角坐标轴x轴之间的角度）和曲率 kappa \n
    ref: https://github.com/6Lackiu/EMplanner_Carla/blob/4cb40d5ca04af8c49f3f7dd6b6966fa70bb7dc2d/planner/planning_utils.py#L185
    :param path_points: 曲线上每一点的坐标 [(x0, y0), (x1, y1), ...]
    :return: [[theta0, kappa0], [theta1, kappa1], ...]
    """
    # 原理:
    # theta = arctan(d_y/d_x)
    # kappa = d_theta / d_s
    # d_s = (d_x^2 + d_y^2)^0.5

    # 用割线斜率近似表示切线斜率，参考数值微分知识
    # 采用中点欧拉法来计算每个点处的切线方向角：
    # 当前点与前一个点连成的线段的方向角、和当前点与下一点连成线段的方向角求平均值，作为该点的切线方向

    # TODO 将填补差分的功能提取至单独的函数中

    points_array = np.array(path_points)
    d_xy_ = (
        points_array[1:, :] - points_array[:-1, :]
    )  # 一阶差分，此种写法比 np.diff() 性能高得多
    d_xy = np.empty_like(points_array)  # 定义变量，预分配内存
    # 由于 n 个点差分得到的只有 n-1 个差分结果，所以要在首尾添加重复单元来近似求每个节点的 dx、dy
    d_xy[0, :] = d_xy_[:, :][0]
    d_xy[-1, :] = d_xy_[:, :][-1]
    d_xy[1:-1, :] = (d_xy_[1:, :] + d_xy_[:-1, :]) / 2

    # 计算切线方向角 theta
    theta = np.arctan2(
        d_xy[:, 1], d_xy[:, 0]
    )  # np.arctan2 会将角度限制在 (-pi, pi)之间

    d_theta_ = theta[1:] - theta[:-1]  # 差分，这种写法比np.diff()性能高得多
    d_theta = np.empty_like(theta)
    d_theta[0] = d_theta_[0]
    d_theta[-1] = d_theta_[-1]
    # d_theta[1:-1] = (d_theta_[1:] + d_theta_[:-1]) / 2  # 准确值，但有多值性风险
    d_theta[1:-1] = np.sin(
        (d_theta_[1:] + d_theta_[:-1]) / 2
    )  # 认为 d_theta 是个小量，用 sin(d_theta) 代替 d_theta，避免多值性

    # 计算曲率 kappa
    d_s = np.sqrt(d_xy[:, 0] ** 2 + d_xy[:, 1] ** 2)
    kappa = d_theta / d_s

    result = np.vstack((theta, kappa)).T
    return result


def enhance_reference_line(
    reference_line_node_list: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """
    将仅包含坐标信息的参考线增强为包含坐标、切向角、曲率的参考线 \n
    :param reference_line_node_list: [[x0, y0], [x1, y1], ...] 参考线上的若干离散点
    :return: [[x0, y0, heading0, kappa0], ...]
    """

    heading_kappa = calculate_heading_kappa(reference_line_node_list)
    return np.hstack((reference_line_node_list, heading_kappa))


def calculate_curvature(points: np.ndarray) -> float:
    """
    曲率半径计算函数 \n
    ref: https://github.com/Pjer-zhang/PJCurvature \n
    :param points: 三个点的坐标
    :return: 曲率半径
    """

    x = points[:, 0]
    y = points[:, 1]

    t_a = np.linalg.norm([x[1] - x[0], y[1] - y[0]])
    t_b = np.linalg.norm([x[2] - x[1], y[2] - y[1]])

    m = np.array([[1, -t_a, t_a**2], [1, 0, 0], [1, t_b, t_b**2]])

    a = np.matmul(np.linalg.inv(m), x)
    b = np.matmul(np.linalg.inv(m), y)

    kappa = 2 * (a[2] * b[1] - b[2] * a[1]) / (a[1] ** 2.0 + b[1] ** 2.0) ** 1.5

    return kappa
