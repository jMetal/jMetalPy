# WFG problem family implementation.
# Adapted from pymoo's WFG implementation (Apache 2.0).
# Reference: https://github.com/anyoptimization/pymoo

from __future__ import annotations

from typing import Optional

import numpy as np

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class WFG(FloatProblem):
    def __init__(
        self,
        number_of_variables: Optional[int] = None,
        number_of_objectives: int = 2,
        k: Optional[int] = None,
        l: Optional[int] = None,
    ) -> None:
        super().__init__()

        if k is None and l is None and number_of_variables is None:
            # jMetal defaults for WFG: k=2, l=4, m=2.
            k = 2
            l = 4
            number_of_variables = k + l
        else:
            if k is None:
                k = 2 if number_of_objectives == 2 else 2 * (number_of_objectives - 1)
            if l is None:
                if number_of_variables is None:
                    l = 4
                    number_of_variables = k + l
                else:
                    l = number_of_variables - k
            if number_of_variables is None:
                number_of_variables = k + l

        self._validate(k, l, number_of_objectives, number_of_variables)

        self.k = k
        self.l = l
        self.n_var = number_of_variables
        self.n_obj = number_of_objectives

        self.obj_directions = [self.MINIMIZE] * number_of_objectives
        self.obj_labels = ["$ f_{} $".format(i) for i in range(number_of_objectives)]

        self.lower_bound = [0.0] * number_of_variables
        self.upper_bound = [float(2 * (i + 1)) for i in range(number_of_variables)]

        self._xu = np.asarray(self.upper_bound, dtype=float)
        self._S = np.arange(2, 2 * number_of_objectives + 1, 2, dtype=float)
        self._A = np.ones(number_of_objectives - 1, dtype=float)

    @staticmethod
    def _validate(k: int, l: int, n_obj: int, n_var: int) -> None:
        if n_obj < 2:
            raise ValueError("WFG problems must have two or more objectives.")
        if k % (n_obj - 1) != 0:
            raise ValueError("Position parameter (k) must be divisible by number_of_objectives - 1.")
        if k + l < n_obj:
            raise ValueError("Sum of distance and position parameters must be >= number_of_objectives.")
        if k + l != n_var:
            raise ValueError("Number of variables must equal k + l.")

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = np.asarray(solution.variables, dtype=float)
        f = self._evaluate(x)
        solution.objectives = [float(v) for v in f]
        return solution

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def _post(self, t: np.ndarray, a: np.ndarray) -> np.ndarray:
        x = []
        for i in range(t.shape[1] - 1):
            x.append(np.maximum(t[:, -1], a[i]) * (t[:, i] - 0.5) + 0.5)
        x.append(t[:, -1])
        return np.column_stack(x)

    def _calculate(self, x: np.ndarray, s: np.ndarray, h: list[np.ndarray]) -> np.ndarray:
        return x[:, -1][:, None] + s * np.column_stack(h)

    def name(self) -> str:
        return "WFG"


class WFG1(WFG):
    @staticmethod
    def t1(x: np.ndarray, n: int, k: int) -> np.ndarray:
        y = x.copy()
        y[:, k:n] = _transformation_shift_linear(y[:, k:n], 0.35)
        return y

    @staticmethod
    def t2(x: np.ndarray, n: int, k: int) -> np.ndarray:
        y = x.copy()
        y[:, k:n] = _transformation_bias_flat(y[:, k:n], 0.8, 0.75, 0.85)
        return y

    @staticmethod
    def t3(x: np.ndarray, n: int) -> np.ndarray:
        y = x.copy()
        y[:, :n] = _transformation_bias_poly(y[:, :n], 0.02)
        return y

    @staticmethod
    def t4(x: np.ndarray, m_obj: int, n: int, k: int) -> np.ndarray:
        w = np.arange(2, 2 * n + 1, 2, dtype=float)
        gap = k // (m_obj - 1)
        t = []
        for obj_index in range(1, m_obj):
            start = (obj_index - 1) * gap
            end = obj_index * gap
            t.append(_reduction_weighted_sum(x[:, start:end], w[start:end]))
        t.append(_reduction_weighted_sum(x[:, k:n], w[k:n]))
        return np.column_stack(t)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        y = x[None, :] / self._xu
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG1.t2(y, self.n_var, self.k)
        y = WFG1.t3(y, self.n_var)
        y = WFG1.t4(y, self.n_obj, self.n_var, self.k)
        y = self._post(y, self._A)

        h = [_shape_convex(y[:, :-1], m + 1) for m in range(self.n_obj - 1)]
        h.append(_shape_mixed(y[:, 0], alpha=1.0, A=5.0))

        return self._calculate(y, self._S, h)[0]

    def name(self) -> str:
        return "WFG1"


class WFG2(WFG):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        validate_wfg2_wfg3(self.l)

    @staticmethod
    def t2(x: np.ndarray, n: int, k: int) -> np.ndarray:
        y = [x[:, i] for i in range(k)]

        l = n - k
        ind_non_sep = k + l // 2

        i = k + 1
        while i <= ind_non_sep:
            head = k + 2 * (i - k) - 2
            tail = k + 2 * (i - k)
            y.append(_reduction_non_sep(x[:, head:tail], 2))
            i += 1

        return np.column_stack(y)

    @staticmethod
    def t3(x: np.ndarray, m_obj: int, n: int, k: int) -> np.ndarray:
        ind_r_sum = k + (n - k) // 2
        gap = k // (m_obj - 1)

        t = [_reduction_weighted_sum_uniform(x[:, (idx - 1) * gap: idx * gap]) for idx in range(1, m_obj)]
        t.append(_reduction_weighted_sum_uniform(x[:, k:ind_r_sum]))

        return np.column_stack(t)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        y = x[None, :] / self._xu
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG2.t2(y, self.n_var, self.k)
        y = WFG2.t3(y, self.n_obj, self.n_var, self.k)
        y = self._post(y, self._A)

        h = [_shape_convex(y[:, :-1], m + 1) for m in range(self.n_obj - 1)]
        h.append(_shape_disconnected(y[:, 0], alpha=1.0, beta=1.0, A=5.0))

        return self._calculate(y, self._S, h)[0]

    def name(self) -> str:
        return "WFG2"


class WFG3(WFG):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        validate_wfg2_wfg3(self.l)
        if self._A.size > 1:
            self._A[1:] = 0.0

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        y = x[None, :] / self._xu
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG2.t2(y, self.n_var, self.k)
        y = WFG2.t3(y, self.n_obj, self.n_var, self.k)
        y = self._post(y, self._A)

        h = [_shape_linear(y[:, :-1], m + 1) for m in range(self.n_obj)]

        return self._calculate(y, self._S, h)[0]

    def name(self) -> str:
        return "WFG3"


class WFG4(WFG):
    @staticmethod
    def t1(x: np.ndarray) -> np.ndarray:
        return _transformation_shift_multi_modal(x, 30.0, 10.0, 0.35)

    @staticmethod
    def t2(x: np.ndarray, m_obj: int, k: int) -> np.ndarray:
        gap = k // (m_obj - 1)
        t = [_reduction_weighted_sum_uniform(x[:, (idx - 1) * gap: idx * gap]) for idx in range(1, m_obj)]
        t.append(_reduction_weighted_sum_uniform(x[:, k:]))
        return np.column_stack(t)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        y = x[None, :] / self._xu
        y = WFG4.t1(y)
        y = WFG4.t2(y, self.n_obj, self.k)
        y = self._post(y, self._A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        return self._calculate(y, self._S, h)[0]

    def name(self) -> str:
        return "WFG4"


class WFG5(WFG):
    @staticmethod
    def t1(x: np.ndarray) -> np.ndarray:
        return _transformation_param_deceptive(x, A=0.35, B=0.001, C=0.05)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        y = x[None, :] / self._xu
        y = WFG5.t1(y)
        y = WFG4.t2(y, self.n_obj, self.k)
        y = self._post(y, self._A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        return self._calculate(y, self._S, h)[0]

    def name(self) -> str:
        return "WFG5"


class WFG6(WFG):
    @staticmethod
    def t2(x: np.ndarray, m_obj: int, n: int, k: int) -> np.ndarray:
        gap = k // (m_obj - 1)
        t = [_reduction_non_sep(x[:, (idx - 1) * gap: idx * gap], gap) for idx in range(1, m_obj)]
        t.append(_reduction_non_sep(x[:, k:], n - k))
        return np.column_stack(t)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        y = x[None, :] / self._xu
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG6.t2(y, self.n_obj, self.n_var, self.k)
        y = self._post(y, self._A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        return self._calculate(y, self._S, h)[0]

    def name(self) -> str:
        return "WFG6"


class WFG7(WFG):
    @staticmethod
    def t1(x: np.ndarray, k: int) -> np.ndarray:
        y = x.copy()
        for i in range(k):
            aux = _reduction_weighted_sum_uniform(y[:, i + 1 :])
            y[:, i] = _transformation_param_dependent(y[:, i], aux)
        return y

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        y = x[None, :] / self._xu
        y = WFG7.t1(y, self.k)
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG4.t2(y, self.n_obj, self.k)
        y = self._post(y, self._A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        return self._calculate(y, self._S, h)[0]

    def name(self) -> str:
        return "WFG7"


class WFG8(WFG):
    @staticmethod
    def t1(x: np.ndarray, n: int, k: int) -> np.ndarray:
        ret = []
        for i in range(k, n):
            aux = _reduction_weighted_sum_uniform(x[:, :i])
            ret.append(_transformation_param_dependent(x[:, i], aux, A=0.98 / 49.98, B=0.02, C=50.0))
        return np.column_stack(ret)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        y = x[None, :] / self._xu
        y[:, self.k : self.n_var] = WFG8.t1(y, self.n_var, self.k)
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG4.t2(y, self.n_obj, self.k)
        y = self._post(y, self._A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        return self._calculate(y, self._S, h)[0]

    def name(self) -> str:
        return "WFG8"


class WFG9(WFG):
    @staticmethod
    def t1(x: np.ndarray, n: int) -> np.ndarray:
        ret = []
        for i in range(0, n - 1):
            aux = _reduction_weighted_sum_uniform(x[:, i + 1 :])
            ret.append(_transformation_param_dependent(x[:, i], aux))
        return np.column_stack(ret)

    @staticmethod
    def t2(x: np.ndarray, n: int, k: int) -> np.ndarray:
        a = [_transformation_shift_deceptive(x[:, i], 0.35, 0.001, 0.05) for i in range(k)]
        b = [_transformation_shift_multi_modal(x[:, i], 30.0, 95.0, 0.35) for i in range(k, n)]
        return np.column_stack(a + b)

    @staticmethod
    def t3(x: np.ndarray, m_obj: int, n: int, k: int) -> np.ndarray:
        gap = k // (m_obj - 1)
        t = [_reduction_non_sep(x[:, (idx - 1) * gap: idx * gap], gap) for idx in range(1, m_obj)]
        t.append(_reduction_non_sep(x[:, k:], n - k))
        return np.column_stack(t)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        y = x[None, :] / self._xu
        y[:, : self.n_var - 1] = WFG9.t1(y, self.n_var)
        y = WFG9.t2(y, self.n_var, self.k)
        y = WFG9.t3(y, self.n_obj, self.n_var, self.k)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        return self._calculate(y, self._S, h)[0]

    def name(self) -> str:
        return "WFG9"


def _transformation_shift_linear(value: np.ndarray, shift: float = 0.35) -> np.ndarray:
    ret = np.fabs(value - shift) / np.fabs(np.floor(shift - value) + shift)
    return correct_to_01(ret)


def _transformation_shift_deceptive(y: np.ndarray, A: float = 0.35, B: float = 0.005, C: float = 0.05) -> np.ndarray:
    tmp1 = np.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
    tmp2 = np.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
    ret = 1.0 + (np.fabs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B)
    return correct_to_01(ret)


def _transformation_shift_multi_modal(y: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    tmp1 = np.fabs(y - C) / (2.0 * (np.floor(C - y) + C))
    tmp2 = (4.0 * A + 2.0) * np.pi * (0.5 - tmp1)
    ret = (1.0 + np.cos(tmp2) + 4.0 * B * np.power(tmp1, 2.0)) / (B + 2.0)
    return correct_to_01(ret)


def _transformation_bias_flat(y: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    ret = a + np.minimum(0, np.floor(y - b)) * (a * (b - y) / b) - np.minimum(0, np.floor(c - y)) * (
        (1.0 - a) * (y - c) / (1.0 - c)
    )
    return correct_to_01(ret)


def _transformation_bias_poly(y: np.ndarray, alpha: float) -> np.ndarray:
    return correct_to_01(y ** alpha)


def _transformation_param_dependent(
    y: np.ndarray, y_deg: np.ndarray, A: float = 0.98 / 49.98, B: float = 0.02, C: float = 50.0
) -> np.ndarray:
    aux = A - (1.0 - 2.0 * y_deg) * np.fabs(np.floor(0.5 - y_deg) + A)
    ret = np.power(y, B + (C - B) * aux)
    return correct_to_01(ret)


def _transformation_param_deceptive(y: np.ndarray, A: float = 0.35, B: float = 0.001, C: float = 0.05) -> np.ndarray:
    tmp1 = np.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
    tmp2 = np.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
    ret = 1.0 + (np.fabs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B)
    return correct_to_01(ret)


def _reduction_weighted_sum(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    return correct_to_01(np.dot(y, w) / w.sum())


def _reduction_weighted_sum_uniform(y: np.ndarray) -> np.ndarray:
    return correct_to_01(y.mean(axis=1))


def _reduction_non_sep(y: np.ndarray, A: int) -> np.ndarray:
    n, m = y.shape
    val = np.ceil(A / 2.0)

    num = np.zeros(n)
    for j in range(m):
        num += y[:, j]
        for k in range(A - 1):
            num += np.fabs(y[:, j] - y[:, (1 + j + k) % m])

    denom = m * val * (1.0 + 2.0 * A - 2 * val) / A

    return correct_to_01(num / denom)


def _shape_concave(x: np.ndarray, m: int) -> np.ndarray:
    M = x.shape[1]
    if m == 1:
        ret = np.prod(np.sin(0.5 * x[:, :M] * np.pi), axis=1)
    elif 1 < m <= M:
        ret = np.prod(np.sin(0.5 * x[:, : M - m + 1] * np.pi), axis=1)
        ret *= np.cos(0.5 * x[:, M - m + 1] * np.pi)
    else:
        ret = np.cos(0.5 * x[:, 0] * np.pi)
    return correct_to_01(ret)


def _shape_convex(x: np.ndarray, m: int) -> np.ndarray:
    M = x.shape[1]
    if m == 1:
        ret = np.prod(1.0 - np.cos(0.5 * x[:, :M] * np.pi), axis=1)
    elif 1 < m <= M:
        ret = np.prod(1.0 - np.cos(0.5 * x[:, : M - m + 1] * np.pi), axis=1)
        ret *= 1.0 - np.sin(0.5 * x[:, M - m + 1] * np.pi)
    else:
        ret = 1.0 - np.sin(0.5 * x[:, 0] * np.pi)
    return correct_to_01(ret)


def _shape_linear(x: np.ndarray, m: int) -> np.ndarray:
    M = x.shape[1]
    if m == 1:
        ret = np.prod(x, axis=1)
    elif 1 < m <= M:
        ret = np.prod(x[:, : M - m + 1], axis=1)
        ret *= 1.0 - x[:, M - m + 1]
    else:
        ret = 1.0 - x[:, 0]
    return correct_to_01(ret)


def _shape_mixed(x: np.ndarray, A: float = 5.0, alpha: float = 1.0) -> np.ndarray:
    aux = 2.0 * A * np.pi
    ret = np.power(1.0 - x - (np.cos(aux * x + 0.5 * np.pi) / aux), alpha)
    return correct_to_01(ret)


def _shape_disconnected(x: np.ndarray, alpha: float = 1.0, beta: float = 1.0, A: float = 5.0) -> np.ndarray:
    aux = np.cos(A * np.pi * x ** beta)
    return correct_to_01(1.0 - x ** alpha * aux ** 2)


def validate_wfg2_wfg3(l: int) -> None:
    if l % 2 != 0:
        raise ValueError("In WFG2/WFG3 the distance-related parameter (l) must be divisible by 2.")


def correct_to_01(x: np.ndarray, epsilon: float = 1.0e-10) -> np.ndarray:
    y = np.asarray(x, dtype=float)
    y = np.where((y < 0.0) & (y >= -epsilon), 0.0, y)
    y = np.where((y > 1.0) & (y <= 1.0 + epsilon), 1.0, y)
    return y
