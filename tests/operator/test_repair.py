"""Unit tests for repair operator classes."""
import numpy as np

from jmetal.operator.repair import (
    ClampFloatRepair,
    FloatRepairOperator,
    IntegerRepairOperator,
    NoOpRepair,
    ensure_float_repair,
)


def test_clamp_float_repair_scalar():
    repair = ClampFloatRepair()
    assert repair.repair_scalar(1.5, 0.0, 1.0) == 1.0
    assert repair.repair_scalar(-0.2, 0.0, 1.0) == 0.0
    assert repair.repair_scalar(0.5, 0.0, 1.0) == 0.5


def test_clamp_float_repair_vector():
    repair = ClampFloatRepair()
    vals = [1.5, -0.2, 0.5]
    lbs = [0.0, 0.0, 0.0]
    ubs = [1.0, 1.0, 1.0]
    out = repair.repair_vector(vals, lbs, ubs)
    assert np.allclose(out, np.array([1.0, 0.0, 0.5]))


def test_ensure_float_repair_wraps_callable():
    def my_repair(v, lb, ub):
        return min(max(lb, v), ub)

    repaired = ensure_float_repair(my_repair)
    # Should provide repair_scalar and repair_vector
    assert isinstance(repaired, FloatRepairOperator)
    assert repaired.repair_scalar(1.2, 0.0, 1.0) == 1.0
    vec = repaired.repair_vector([1.2, -1.0], [0.0, 0.0], [1.0, 1.0])
    assert np.allclose(vec, np.array([1.0, 0.0]))


def test_noop_repair():
    noop = NoOpRepair()
    assert noop.repair_scalar(5.0, 0.0, 1.0) == 5.0
    assert np.allclose(noop.repair_vector([1, 2, 3], [0, 0, 0], [10, 10, 10]), np.array([1, 2, 3]))


def test_integer_repair():
    ir = IntegerRepairOperator()
    assert isinstance(ir.repair_scalar(2.7, 0, 5), int)
    assert ir.repair_scalar(10.9, 0, 5) == 5
    vec = ir.repair_vector([1.2, 4.9, 7.1], [0, 0, 0], [5, 5, 5])
    assert np.array_equal(vec, np.array([1, 5, 5]))


def test_bound_swap_and_reflective_and_random_uniform():
    from jmetal.operator.repair import RandomUniformRepair, ReflectiveRepair, BoundSwapRepair

    # BoundSwapRepair scalar and vector
    bs = BoundSwapRepair()
    assert bs.repair_scalar(1.5, 0.0, 1.0) == 0.0
    assert bs.repair_scalar(-0.5, 0.0, 1.0) == 1.0
    arr = np.array([-2.0, 0.5, 3.0])
    lbs = np.array([0.0, 0.0, 0.0])
    ubs = np.array([1.0, 1.0, 1.0])
    out_bs = bs.repair_vector(arr, lbs, ubs)
    assert np.array_equal(out_bs, np.array([1.0, 0.5, 0.0]))

    # ReflectiveRepair scalar and vector
    rf = ReflectiveRepair()
    assert rf.repair_scalar(12.0, 0.0, 10.0) == 8.0
    assert rf.repair_scalar(-3.0, 0.0, 10.0) == 3.0
    assert rf.repair_scalar(25.0, 0.0, 10.0) == 5.0
    arr = np.array([-3.0, 5.0, 12.0, 25.0])
    lbs = np.array([0.0, 0.0, 0.0, 0.0])
    ubs = np.array([10.0, 10.0, 10.0, 10.0])
    out_rf = rf.repair_vector(arr, lbs, ubs)
    assert np.array_equal(out_rf, np.array([3.0, 5.0, 8.0, 5.0]))

    # RandomUniformRepair reproducibility
    seed = 42
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)
    ru1 = RandomUniformRepair(rng=rng1)
    ru2 = RandomUniformRepair(rng=rng2)
    s1 = ru1.repair_scalar(2.0, 0.0, 1.0)
    s2 = ru2.repair_scalar(2.0, 0.0, 1.0)
    assert 0.0 <= s1 <= 1.0
    assert s1 == s2
    arr = np.array([-1.0, 0.5, 2.0, 3.0])
    lbs = np.array([0.0, 0.0, 0.0, 0.0])
    ubs = np.array([1.0, 1.0, 1.0, 1.0])
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)
    ru1 = RandomUniformRepair(rng=rng1)
    ru2 = RandomUniformRepair(rng=rng2)
    out1 = ru1.repair_vector(arr, lbs, ubs)
    out2 = ru2.repair_vector(arr, lbs, ubs)
    assert np.array_equal(out1, out2)
    assert np.all((out1 >= 0.0) & (out1 <= 1.0))
