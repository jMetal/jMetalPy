import numpy as np
import random
from typing import cast, Callable
from jmetal.core.solution import FloatSolution, IntegerSolution
from jmetal.operator.mutation import PolynomialMutation, IntegerPolynomialMutation, NonUniformMutation, UniformMutation
from jmetal.operator.repair import ensure_float_repair, ensure_integer_repair


def make_float_solution(n, lower=0.0, upper=1.0, values=None):
    lbs = [lower] * n
    ubs = [upper] * n
    s = FloatSolution(lbs, ubs, 1)
    if values is None:
        s._variables = [(lower + upper) / 2.0] * n
    else:
        s._variables = list(values)
    return s


def make_int_solution(n, lower=0, upper=10, values=None):
    lbs = [lower] * n
    ubs = [upper] * n
    s = IntegerSolution(lbs, ubs, 1)
    if values is None:
        s._variables = [int((lower + upper) // 2)] * n
    else:
        s._variables = list(values)
    return s


def test_polynomial_mutation_uses_callable_repair():
    # Force mutation with probability=1.0
    random.seed(0)
    s = make_float_solution(1, 0.0, 1.0, [0.5])

    # Repair callable that forces lower bound
    def to_lower(v, lb, ub):
        return lb

    op = PolynomialMutation(probability=1.0, distribution_index=20.0, repair_operator=to_lower)
    op.execute(s)
    assert s._variables[0] == 0.0


def test_integer_polynomial_mutation_uses_integer_repair():
    random.seed(0)
    s = make_int_solution(1, 0, 5, [2])

    # Repair callable that forces upper bound
    def to_upper(v, lb, ub):
        return ub

    op = IntegerPolynomialMutation(probability=1.0, distribution_index=20.0, repair_operator=to_upper)
    op.execute(s)
    assert s._variables[0] == 5


def test_nonuniform_mutation_uses_repair_operator():
    random.seed(0)
    s = make_float_solution(1, 0.0, 1.0, [0.5])

    # Repair callable that forces upper bound
    def to_upper(v, lb, ub):
        return ub

    op = NonUniformMutation(probability=1.0, perturbation=1.0, max_iterations=100, repair_operator=to_upper)
    op.set_current_iteration(10)
    op.execute(s)
    assert s._variables[0] == 1.0


def test_uniform_mutation_accepts_repair_instance_and_callable():
    n = 5
    seed = 42
    rng = np.random.default_rng(seed)

    s1 = make_float_solution(n, 0.0, 1.0, list(np.linspace(0.2, 0.8, n)))
    s2 = make_float_solution(n, 0.0, 1.0, list(np.linspace(0.2, 0.8, n)))

    # repair callable forces lower bound
    def to_lower(v, lb, ub):
        return lb

    # Using callable
    op_callable = UniformMutation(probability=1.0, perturbation=1.0, repair_operator=to_lower, rng=rng)
    op_callable.execute(s1)

    # Using instance
    repair_inst = ensure_float_repair(to_lower)
    rng2 = np.random.default_rng(seed)
    # cast to satisfy type checkers in tests (repair instance is accepted at runtime)
    op_inst = UniformMutation(probability=1.0, perturbation=1.0, repair_operator=cast(Callable, repair_inst), rng=rng2)
    op_inst.execute(s2)

    assert all(v == 0.0 for v in s1._variables)
    assert all(v == 0.0 for v in s2._variables)
