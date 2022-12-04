"""
计算曲率半径
"""

import numpy as np


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
