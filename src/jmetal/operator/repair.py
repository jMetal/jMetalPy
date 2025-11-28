from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np

NumericArray = Union[np.ndarray, list, tuple]


class FloatRepairOperator:
    """Base interface for repair operators that work on continuous (float) variables.

    Implementations should provide two APIs:
    - `repair_scalar(value, lb, ub)` repairs a single float value.
    - `repair_vector(values, lbs, ubs)` repairs arrays element-wise and returns a
      numpy array.

    A callable adapter (`ensure_float_repair`) is provided so that existing code
    that passes simple scalar callables continues to work. The default
    `repair_vector` implementation applies `repair_scalar` element-wise.
    """

    def repair_scalar(self, value: float, lower_bound: float, upper_bound: float) -> float:
        raise NotImplementedError()

    def repair_vector(
        self,
        values: NumericArray,
        lower_bounds: NumericArray,
        upper_bounds: NumericArray,
    ) -> np.ndarray:
        # Default implementation: call repair_scalar for each element.
        vals = np.asarray(values, dtype=float)
        lbs = np.asarray(lower_bounds, dtype=float)
        ubs = np.asarray(upper_bounds, dtype=float)
        out = np.empty_like(vals)
        for i in range(vals.size):
            out[i] = self.repair_scalar(vals[i], lbs[i], ubs[i])
        return out

    def __call__(
        self,
        values: Union[float, NumericArray],
        lower_bounds: Union[float, NumericArray],
        upper_bounds: Union[float, NumericArray],
    ) -> Union[float, np.ndarray]:
        """Convenience dispatcher: if `values` is array-like, call `repair_vector`,
        otherwise call `repair_scalar`.
        """
        if isinstance(values, (list, tuple, np.ndarray)):
            return self.repair_vector(values, lower_bounds, upper_bounds)
        return self.repair_scalar(float(values), float(lower_bounds), float(upper_bounds))


class ClampFloatRepair(FloatRepairOperator):
    """Default clamp-to-bounds repair for float variables.

    This implementation reproduces the common `min(max(x, lb), ub)` behavior and
    provides an optimized `repair_vector` using `numpy.clip`.
    """

    def repair_scalar(self, value: float, lower_bound: float, upper_bound: float) -> float:
        return float(max(lower_bound, min(upper_bound, value)))

    def repair_vector(self, values: NumericArray, lower_bounds: NumericArray, upper_bounds: NumericArray) -> np.ndarray:
        vals = np.asarray(values, dtype=float)
        lbs = np.asarray(lower_bounds, dtype=float)
        ubs = np.asarray(upper_bounds, dtype=float)
        # np.clip supports broadcasting and is efficient for vectors
        return np.clip(vals, lbs, ubs)


class IntegerRepairOperator:
    """Repair operator for integer-valued variables.

    Default behavior: round to the nearest integer and clamp to bounds.
    """

    def repair_scalar(self, value: float, lower_bound: int, upper_bound: int) -> int:
        v = int(round(value))
        return int(max(lower_bound, min(upper_bound, v)))

    def repair_vector(self, values: NumericArray, lower_bounds: NumericArray, upper_bounds: NumericArray) -> np.ndarray:
        vals = np.rint(np.asarray(values, dtype=float)).astype(int)
        lbs = np.asarray(lower_bounds, dtype=int)
        ubs = np.asarray(upper_bounds, dtype=int)
        return np.minimum(np.maximum(vals, lbs), ubs)


class NoOpRepair:
    """No-op repair operator: returns inputs unchanged.

    Useful for solution types that do not require repair (binary, permutation),
    or as a placeholder in tests.
    """

    def repair_scalar(self, value, *_):
        return value

    def repair_vector(self, values: NumericArray, *_):
        return np.asarray(values)


class RandomUniformRepair(FloatRepairOperator):
    """Repair operator that replaces out-of-bounds values by a uniform sample
    inside the provided bounds. Uses a NumPy Generator for reproducibility.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self._rng = rng or np.random.default_rng()

    def repair_scalar(self, value: float, lower_bound: float, upper_bound: float) -> float:
        if lower_bound >= upper_bound:
            return float(lower_bound)
        if lower_bound <= value <= upper_bound:
            return float(value)
        return float(self._rng.uniform(lower_bound, upper_bound))

    def repair_vector(self, values: NumericArray, lower_bounds: NumericArray, upper_bounds: NumericArray) -> np.ndarray:
        vals = np.asarray(values, dtype=float)
        lbs = np.asarray(lower_bounds, dtype=float)
        ubs = np.asarray(upper_bounds, dtype=float)
        out = vals.copy()
        mask_low = vals < lbs
        mask_high = vals > ubs
        mask = mask_low | mask_high
        if mask.any():
            # draw uniform samples for masked positions
            samples = self._rng.uniform(lbs[mask], ubs[mask])
            out[mask] = samples
        return out


class ReflectiveRepair(FloatRepairOperator):
    """Reflective (mirror) repair: values outside bounds are reflected back
    into the interval. Repeated reflections handled via modulo arithmetic.
    """

    def repair_scalar(self, value: float, lower_bound: float, upper_bound: float) -> float:
        lb = float(lower_bound)
        ub = float(upper_bound)
        if lb >= ub:
            return lb
        rng = ub - lb
        if rng == 0:
            return lb
        # offset from lb
        off = value - lb
        # map into [0, 2*rng) then reflect into [0, rng]
        m = off % (2 * rng)
        if m <= rng:
            return lb + m
        return ub - (m - rng)

    def repair_vector(self, values: NumericArray, lower_bounds: NumericArray, upper_bounds: NumericArray) -> np.ndarray:
        vals = np.asarray(values, dtype=float)
        lbs = np.asarray(lower_bounds, dtype=float)
        ubs = np.asarray(upper_bounds, dtype=float)
        out = np.empty_like(vals)
        # vectorized reflection using modulo on each element
        ranges = ubs - lbs
        # Handle degenerate ranges
        zero_mask = ranges == 0
        if zero_mask.any():
            out[zero_mask] = lbs[zero_mask]
        nonzero = ~zero_mask
        if nonzero.any():
            off = vals[nonzero] - lbs[nonzero]
            two_r = 2 * ranges[nonzero]
            m = np.mod(off, two_r)
            mask = m <= ranges[nonzero]
            out_nonzero = np.empty_like(m)
            out_nonzero[mask] = lbs[nonzero][mask] + m[mask]
            out_nonzero[~mask] = ubs[nonzero][~mask] - (m[~mask] - ranges[nonzero][~mask])
            out[nonzero] = out_nonzero
        return out


class BoundSwapRepair(FloatRepairOperator):
    """If value exceeds upper bound assign lower bound, and viceversa.

    This is an aggressive repair that 'jumps' the value to the opposite bound.
    """

    def repair_scalar(self, value: float, lower_bound: float, upper_bound: float) -> float:
        if value > upper_bound:
            return float(lower_bound)
        if value < lower_bound:
            return float(upper_bound)
        return float(value)

    def repair_vector(self, values: NumericArray, lower_bounds: NumericArray, upper_bounds: NumericArray) -> np.ndarray:
        vals = np.asarray(values, dtype=float)
        lbs = np.asarray(lower_bounds, dtype=float)
        ubs = np.asarray(upper_bounds, dtype=float)
        out = vals.copy()
        out[vals > ubs] = lbs[vals > ubs]
        out[vals < lbs] = ubs[vals < lbs]
        return out


def ensure_float_repair(repair: Optional[Union[FloatRepairOperator, Callable]]) -> FloatRepairOperator:
    """Normalize a `repair` argument into a `FloatRepairOperator` instance.

    Rules:
    - If `repair` is None: return `ClampFloatRepair()` (default clamp behavior).
    - If `repair` is already a `FloatRepairOperator`: return it unchanged.
    - If `repair` is a callable: wrap it in an adapter that implements
      `repair_scalar` and inherits the default `repair_vector` behavior.
    """
    if repair is None:
        return ClampFloatRepair()

    if isinstance(repair, FloatRepairOperator):
        return repair

    if callable(repair):
        func = repair

        class _CallableRepair(FloatRepairOperator):
            def repair_scalar(self, value: float, lower_bound: float, upper_bound: float) -> float:
                return float(func(value, lower_bound, upper_bound))

            # inherits repair_vector which calls repair_scalar element-wise

        return _CallableRepair()

    raise TypeError("repair must be None, callable or FloatRepairOperator")


def ensure_integer_repair(repair: Optional[Union[IntegerRepairOperator, Callable]]) -> IntegerRepairOperator:
    if repair is None:
        return IntegerRepairOperator()
    if isinstance(repair, IntegerRepairOperator):
        return repair
    if callable(repair):
        func = repair

        class _CallableIntRepair(IntegerRepairOperator):
            def repair_scalar(self, value: float, lower_bound: int, upper_bound: int) -> int:
                return int(func(value, lower_bound, upper_bound))

        return _CallableIntRepair()

    raise TypeError("repair must be None, callable or IntegerRepairOperator")
