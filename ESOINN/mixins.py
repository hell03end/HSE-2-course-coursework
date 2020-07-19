from collections import Callable, Collection
import numpy as np


class LoggerMixin(object):
    def __init__(self, *args, **kwargs):
        from .config import Logger, LEVELS
        params = {}
        params['level'] = kwargs.pop("level", LEVELS['DEBUG'])
        fmt = kwargs.pop("format", None)
        if fmt:
            params['format'] = fmt
        self._logger = Logger(str(self.__class__), **params)
        super(LoggerMixin, self).__init__(*args, **kwargs)


class BaseMetaclass(type):
    def __new__(meta_class, name: str, bases: Collection, class_dict: dict):
        # normal class constructor
        new_class = super(BaseMetaclass, meta_class).__new__(
            meta_class, name, bases, class_dict
        )
        # recursively inherit docstrings
        for attr_name, attr in class_dict.items():
            if not isinstance(attr, Callable) or attr.__doc__:
                continue
            for base_class in new_class.mro():  # method resolution order
                if not hasattr(base_class, attr_name):
                    continue
                base_func = getattr(base_class, attr_name)
                if base_func.__doc__:
                    attr.__doc__ = base_func.__doc__
                    break
        return new_class

    def __call__(cls, *args, **kwargs):
        self = super(BaseMetaclass, cls).__call__(*args, **kwargs)
        return self


class BaseClass(BaseMetaclass("BaseNetwork", (), {})):
    """Generic class for all objects"""

    def __init__(self, *args, **kwargs):
        super(BaseClass, self).__init__(*args, **kwargs)


class BaseNetwork(BaseClass, LoggerMixin):
    """Generic class for NN"""

    def __init__(self, *args, **kwargs):
        self._cache = None
        super(BaseNetwork, self).__init__(*args, **kwargs)


class BaseNode(BaseClass, LoggerMixin):
    """Generic class for NN node"""

    def __init__(self, feature_vector: (list, set, tuple), *args, **kwargs):
        self._fv = np.array(feature_vector, dtype=float)  # feature vector
        self._acc = 0  # local accumulated number of samples (being winner)
        self._cls = -1  # subclass id
        self._points = 0.0  # total points
        self._dens = 0.0  # density
        super(BaseNode, self).__init__(*args, **kwargs)

    @property
    def features(self) -> np.ndarray:
        return self._fv.copy()

    @property
    def subclass(self) -> int:
        return self._cls

    @subclass.setter
    def subclass(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError(
                "subclass should be int not {}".format(type(value))
            )
        self._cls = value if value > 0 else -1

    @property
    def accumulated_signals(self) -> int:
        return self._acc

    @property
    def points(self) -> float:
        return self._points

    @points.setter
    def points(self, value: float) -> None:
        if value < 0:
            raise ValueError("only positive numbers allowed")
        self._points += value

    @property
    def density(self) -> float:
        return self._dens

    def update_accumulated_signals(self, n: int=1) -> None:
        if n < 1:
            raise ValueError("only positive numbers allowed")
        if not isinstance(n, int):
            raise ValueError("n should be int not {}".format(type(n)))
        self._acc += n

    def reset_points(self) -> None:
        self._points = 0

    def update_density(self, coeff: float=None) -> None:
        if coeff:
            self._dens = self._points / coeff
        else:
            self._dens = self._points / self._acc

    def update_feature_vector(self, signal: np.array, coeff: float) -> None:
        self.__weights += coeff * (signal - self.__weights)

    @staticmethod
    def _calc_dist(x: np.ndarray, y: np.ndarray, method: str) -> float:
        metrics = {
            "euclidean": lambda x, y: np.sqrt(
                np.sum(np.square(np.subtract(x, y)))
            ),
            "taxicab": lambda x, y: np.sum(np.abs(np.subtract(x, y))),
            "cosine": lambda x, y: np.divide(
                np.sum(np.multiply(x, y)),
                np.sqrt(np.multiply(
                    np.sum(np.square(x)), np.sum(np.square(y))
                ))
            )
        }
        return metrics[method](x, y)

    def dist_to(self, node: BaseNode, metrics: (str, Callable)) -> float:
        if isinstance(metrics, str):
            return self._calc_dist(self._fv, node._fv, metrics)
        return metrics(self._fv, node._fv)

    @property
    def T(self) -> np.ndarray:
        return self._fv.T

    def dot(self, value: np.ndarray) -> (int, float, np.ndarray):
        return self._fv.dot(value)

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__, self._fv)

    def __str__(self) -> str:
        return "<{} at {}>::fv{:^30}::acc={:10}::cls={:10}" \
            "::points={:13.10}::dens={:13.10}".format(
                self.__class__,
                hex(id(self)),
                str(self._fv),
                str(self._acc),
                str(self._cls),
                float(self._points),
                float(self._dens)
            )

    def __bool__(self) -> str:
        return self._fv is not None

    def __getitem__(self, idx) -> float:
        return self._fv[idx]

    def __len__(self) -> int:
        return self._fv.size

    def __add__(self, value: (float, int, np.ndarray)) -> np.ndarray:
        return np.add(self._fv, value, dtype=float)

    def __sub__(self, value: (float, int, np.ndarray)) -> np.ndarray:
        return np.subtract(self._fv, value, dtype=float)

    def __mul__(self, value: (float, int, np.ndarray)) -> np.ndarray:
        return np.multiply(self._fv, value, dtype=float)

    def __truediv__(self, value: (float, int, np.ndarray)) -> np.ndarray:
        return np.divide(self._fv, value, dtype=float)

    def __floordiv__(self, value: (float, int, np.ndarray)) -> np.ndarray:
        return np.floor_divide(self._fv, value, dtype=float)

    def __mod__(self, value: (float, int, np.ndarray)) -> np.ndarray:
        return np.mod(self._fv, value, dtype=float)

    def __pow__(self, value: (float, int, np.ndarray)) -> np.ndarray:
        return np.power(self._fv, value, dtype=float)

    def __iadd__(self, value: (float, int, np.ndarray)) -> None:
        __ = np.add(self._fv, value, out=self._fv, dtype=float)

    def __isub__(self, value: (float, int, np.ndarray)) -> None:
        __ = np.subtract(self._fv, value, out=self._fv, dtype=float)

    def __imul__(self, value: (float, int, np.ndarray)) -> None:
        __ = np.multiply(self._fv, value, out=self._fv, dtype=float)

    def __itruediv__(self, value: (float, int, np.ndarray)) -> None:
        __ = np.divide(self._fv, value, out=self._fv, dtype=float)

    def __ifloordiv__(self, value: (float, int, np.ndarray)) -> None:
        __ = np.floor_divide(self._fv, value, out=self._fv, dtype=float)

    def __imod__(self, value: (float, int, np.ndarray)) -> None:
        __ = np.mod(self._fv, value, out=self._fv, dtype=float)

    def __ipow__(self, value: (float, int, np.ndarray)) -> None:
        __ = np.power(self._fv, value, out=self._fv, dtype=float)


__all__ = ["BaseNetwork", "BaseNode", "LoggerMixin"]
