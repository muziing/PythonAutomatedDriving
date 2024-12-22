from typing import Optional

import numpy as np
import numpy.typing as npt


class CartesianPath:
    """
    笛卡尔坐标系下的路径，包含 x y heading kappa
    """

    def __init__(self, data_ndarray: npt.NDArray[np.float_]):
        if data_ndarray.shape[1] == FrenetPoint.dimension:
            self.ndarray = data_ndarray
            self.length: int = self.ndarray.shape[0]
        else:
            raise ValueError(
                f"could not broadcast input array from shape "
                f"(,{data_ndarray.shape[1]}) into shape (,4)",
            )

    def __getitem__(self, item):
        return self.ndarray[item]


class FrenetPoint:
    """
    Frenet 坐标系下的点
    """

    def __init__(self, s: float, l: float, dl: float, ddl: float):
        self.s = s
        self.l = l
        self.dl = dl  # dl/ds
        self.ddl = ddl  # d(dl/ds)/ds

    def to_array(self) -> np.ndarray:
        """
        以 numpy.ndarray 形式返回点坐标
        :return: [s, l, dl, ddl]
        """

        return np.array((self.s, self.l, self.dl, self.ddl))

    def __eq__(self, other):
        if self.s == other.s and self.l == other.l:
            return True
        else:
            return False

    def __repr__(self):
        return f"FrenetPoint, s={self.s}, l={self.l}, dl={self.dl}, ddl={self.ddl}"

    dimension: int = 4  # 包含的信息维度


class FrenetPath:
    """
    Frenet 坐标系下由若干点组成的路径，包含 s l dl ddl
    """

    def __init__(
        self,
        *,
        points: Optional[list[FrenetPoint]] = None,
        data_ndarray: Optional[npt.NDArray[np.float_]] = None,
    ):
        if points is not None:
            self.__init_from_list(points)
        elif data_ndarray is not None:
            self.__init_from_ndarray(data_ndarray)
        else:
            raise AttributeError("FrenetPath 初始化失败，需要传入 points data_ndarray 之一")

        self.length: int = self.ndarray.shape[0]

    def __init_from_list(self, points: list[FrenetPoint]):
        self._points_to_ndarray(points)

    def __init_from_ndarray(self, data_ndarray: npt.NDArray[np.float_]):
        if data_ndarray.shape[1] == FrenetPoint.dimension:
            self.ndarray = data_ndarray
        else:
            raise ValueError(
                f"could not broadcast input array from shape "
                f"(,{data_ndarray.shape[1]}) into shape (,{FrenetPoint.dimension})",
            )

    def __getitem__(self, item):
        if type(item) is int:
            # 按 frenet_line[0] 的形式切片，将获得一个 FrenetPoint 实例
            return FrenetPoint(*list(self.ndarray[item, :]))
        else:
            # 按 frenet_line[:, 0] 的形式切片，将获得矩阵的一列
            return self.ndarray[item]

    def _points_to_ndarray(self, points: list[FrenetPoint]) -> None:
        """
        将 FrenetPoint 构成的列表转换到 numpy 矩阵形式存储
        """

        self.ndarray = np.empty((self.length, FrenetPoint.dimension))
        for i in range(self.length):
            self.ndarray[i, :] = points[i].to_array()
