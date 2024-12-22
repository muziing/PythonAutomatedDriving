"""
坐标系变换算法
"""

import math

import numpy as np

from src.utilities.structures import CartesianPath, FrenetPath


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


def cartesian_to_frenet_1d(
    x: np.ndarray,
    y: np.ndarray,
    frenet_path_x: np.ndarray,
    frenet_path_y: np.ndarray,
    proj_x: np.ndarray,
    proj_y: np.ndarray,
    proj_heading: np.ndarray,
    proj_match_point_index: list,
    index2s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    笛卡尔坐标系转 Frenet 坐标系 得到 s, l
    :param x: 需要转换的笛卡尔坐标 x
    :param y: 需要转换的笛卡尔坐标 y
    :param frenet_path_x: Frenet 坐标轴的笛卡尔坐标 x
    :param frenet_path_y: Frenet 坐标轴的笛卡尔坐标 y
    :param proj_x: 投影点笛卡尔坐标 x
    :param proj_y:  投影点笛卡尔坐标 y
    :param proj_heading: 投影点笛卡尔坐标 heading
    :param proj_match_point_index: 投影点坐标在 Frenet 坐标轴离散点的索引
    :param index2s: frenet_path 的 index 与 s 的转换表
    :return: (s, l) Frenet
    """

    frenet = np.append(
        frenet_path_x[proj_match_point_index],
        frenet_path_y[proj_match_point_index],
        axis=1,
    )
    vector_1 = np.append(proj_x, proj_y, axis=1) - frenet
    index = np.array(
        [
            np.array([value + 1, 1])
            if value < len(frenet_path_x)
            else np.array([value - 1, -1])
            for value in proj_match_point_index
        ]
    )
    vector_2 = index[:, 1] * (
        np.append(frenet_path_x[index[:, 0]], frenet_path_y[index[:, 0]], axis=1)
        - frenet
    )
    sign_2 = np.array(
        [np.sign(np.sum(vector_1 * vector_2, axis=1))]
    ).T  # np.sum得到的是一个数组，(n, )
    s = (
        index2s[proj_match_point_index]
        + sign_2 * np.array([np.sqrt(np.sum(vector_1 * vector_1, axis=1))]).T
    )
    n_r = np.array(list(map(lambda val: [-math.sin(val), math.cos(val)], proj_heading)))
    r_h = np.append(x, y, axis=1)
    r_r = np.append(proj_x, proj_y, axis=1)
    l = np.array([np.sum((r_h - r_r) * n_r, axis=1)]).T
    return s, l


def cartesian_to_frenet_2d(
    l: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    proj_heading: np.ndarray,
    proj_kappa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    笛卡尔坐标系转 Frenet 坐标系 得到 s_dot，l_dot，dl
    :param l: Frenet 坐标系中的l
    :param vx: 纵向速度
    :param vy: 横向速度
    :param proj_heading: 投影点的航向角
    :param proj_kappa: 投影点的曲率
    :return: (s_dot, l_dot, dl) s对t的导、l对t的导、l对s的导
    """

    v_h = np.append(vx, vy, axis=1)
    n_r = np.array(map(lambda val: [-math.sin(val), math.cos(val)], proj_heading))
    t_r = np.array(map(lambda val: [math.cos(val), math.sin(val)], proj_heading))
    l_dot = np.array([np.sum(v_h * n_r, axis=1)]).T
    s_dot = np.array([np.sum(v_h * t_r, axis=1)]).T / (1 - proj_kappa * l) + 1e-8
    dl = l_dot / s_dot
    return s_dot, l_dot, dl


def cartesian_to_frenet_3d(
    ax: np.ndarray,
    ay: np.ndarray,
    proj_heading: np.ndarray,
    proj_kappa: np.ndarray,
    l: np.ndarray,
    s_dot: np.ndarray,
    dl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    笛卡尔坐标系转 Frenet 坐标系 得到 s_dot2, l_dot2, ddl
    :param ax: 横向加速度
    :param ay: 纵向加速度
    :param proj_heading: 投影点的航向角
    :param proj_kappa: 投影点的曲率
    :param l: Frenet 坐标系中l
    :param s_dot: Frenet 坐标系中s对t的导
    :param dl: l对s的导
    :return: (s_dot2, l_dot2, ddl) Frenet坐标系中s对t的二阶导、Frenet坐标系中l对t的二阶导、l对s的二阶导
    """

    a_h = np.append(ax, ay, axis=1)
    n_r = np.array(map(lambda val: [-math.sin(val), math.cos(val)], proj_heading))
    t_r = np.array(map(lambda val: [math.cos(val), math.sin(val)], proj_heading))
    l_dot2 = np.array([np.sum(a_h * n_r, axis=1)]).T - proj_kappa * (
        1 - proj_kappa * l
    ) * np.power(s_dot, 2)
    s_dot2 = (1 / (1 - proj_kappa * l)) * (
        np.array([np.sum(a_h * t_r, axis=1)]).T
        + 2 * proj_kappa * dl * np.power(s_dot, 2)
    )
    ddl = (l_dot2 - dl * s_dot2) / np.power(s_dot, 2)

    return s_dot2, l_dot2, ddl


def frenet_to_cartesian(
    path: FrenetPath,
    frenet_path_x: np.ndarray,
    frenet_path_y: np.ndarray,
    frenet_path_heading: np.ndarray,
    frenet_path_kappa: np.ndarray,
    match_index: list,
    index2s: np.ndarray,
) -> CartesianPath:
    """
    Frenet 坐标系转笛卡尔坐标系
    :param path: Frenet 坐标系下的 s l dl ddl
    :param frenet_path_x: 参考线的笛卡尔坐标x
    :param frenet_path_y: 参考线的笛卡尔坐标y
    :param frenet_path_heading: 参考线的航向角
    :param frenet_path_kappa: 参考线的曲率
    :param match_index: s对应的匹配点索引
    :param index2s: s与索引的映射
    :return: (x, y, heading, kappa)
    """

    match_point = np.append(
        frenet_path_x[match_index], frenet_path_y[match_index], axis=1
    )
    match_point_heading = frenet_path_heading[match_index]
    match_point_kappa = frenet_path_kappa[match_index]

    s = path[:, 0]
    l = path[:, 1]
    dl = path[:, 2]
    ddl = path[:, 3]

    ds = s - index2s[match_index]

    match_tor = np.array(
        list(map(lambda val: [math.cos(val), math.sin(val)], match_point_heading))
    )
    proj_point = match_point + ds * match_tor
    proj_heading = match_point_heading + ds * match_point_kappa
    proj_kappa = match_point_kappa
    nor = np.array(list(map(lambda val: [-math.sin(val), math.cos(val)], proj_heading)))

    point = proj_point + l * nor
    heading = proj_heading + np.arctan(dl / (1 - proj_kappa * l))
    kappa = (
        (
            (ddl + proj_kappa * dl * np.tan(heading - proj_heading))
            * np.power(np.cos(heading - proj_heading), 2)
            / (1 - proj_kappa * l)
            + proj_kappa
        )
        * np.cos(heading - proj_heading)
        / (1 - proj_kappa * l)
    )

    cartesian_path = np.empty((point.shape[1], 4))
    cartesian_path[:, 0] = point[:, 0]
    cartesian_path[:, 1] = point[:, 1]
    cartesian_path[:, 2] = heading
    cartesian_path[:, 3] = kappa

    return CartesianPath(cartesian_path)
