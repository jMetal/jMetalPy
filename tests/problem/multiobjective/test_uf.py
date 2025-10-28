import numpy as np
import pytest

from jmetal.problem.multiobjective import uf
from jmetal.core.solution import FloatSolution

@pytest.mark.parametrize("problem_class, n_var, n_obj", [
    (uf.UF1, 30, 2),
    (uf.UF2, 30, 2),
    (uf.UF3, 30, 2),
    (uf.UF4, 30, 2),
    (uf.UF5, 30, 2),
    (uf.UF6, 30, 2),
    (uf.UF7, 30, 2),
    (uf.UF8, 30, 3),
    (uf.UF9, 30, 3),
    (uf.UF10, 30, 3),
])
def test_uf_problem_basic_properties(problem_class, n_var, n_obj):
    problem = problem_class(n_var)
    assert problem.number_of_variables() == n_var
    assert problem.number_of_objectives() == n_obj
    assert problem.number_of_constraints() == 0

@pytest.mark.parametrize("problem_class, n_var, n_obj", [
    (uf.UF1, 30, 2),
    (uf.UF2, 30, 2),
    (uf.UF3, 30, 2),
    (uf.UF4, 30, 2),
    (uf.UF5, 30, 2),
    (uf.UF6, 30, 2),
    (uf.UF7, 30, 2),
    (uf.UF8, 30, 3),
    (uf.UF9, 30, 3),
    (uf.UF10, 30, 3),
])
def test_uf_problem_evaluate_returns_finite_objectives(problem_class, n_var, n_obj):
    problem = problem_class(n_var)
    # Genera una solución válida en el centro del espacio de búsqueda
    variables = [
        (lb + ub) / 2.0 for lb, ub in zip(problem.lower_bound, problem.upper_bound)
    ]
    solution = FloatSolution(problem.lower_bound, problem.upper_bound, n_obj)
    solution.variables = variables
    evaluated = problem.evaluate(solution)
    assert len(evaluated.objectives) == n_obj
    for obj in evaluated.objectives:
        assert np.isfinite(obj)
