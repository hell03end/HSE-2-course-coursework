from collections import Collection, Callable
from copy import deepcopy
import numpy as np
from time import time

from .mixins import BaseNode


class ESOINNNode(BaseNode):
    def __init__(self, feature_vector: (list, tuple, set)):
        super(ESOINNNode, self).__init__(feature_vector)


class ESOINNNodeWithPeriods(BaseNode):
    def __init__(self, feature_vector: (list, tuple, set)):
        super(ESOINNNodeWithPeriods, self).__init__(feature_vector)
        self._period = -1
        self._period_w = 0

    @property
    def periods(self) -> int:
        return self._period

    @property
    def winning_periods(self) -> int:
        return self._period_w

    def update_win_period(self, value: int) -> None:
        if value != self._period:
            self._period = value
            self._period_w += 1

    def update_density(self, coeff: float=None) -> None:
        if coeff:
            self._dens = self._points / coeff
        else:
            self._dens = self._points / self._period_w

    def __repr__(self) -> str:
        return "{}::period={:13.10}::period_w={:13.10}".format(
            super(ESOINNNodeWithPeriods, self).__repr__(),
            float(self._period),
            float(self._period_w)
        )
