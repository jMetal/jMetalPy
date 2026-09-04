"""Tests for AlgorithmState."""

from jmetal.component.algorithm.algorithm_state import AlgorithmState
from jmetal.problem.singleobjective.unconstrained import Sphere


class TestAlgorithmState:
    def _state(self) -> AlgorithmState:
        problem = Sphere(number_of_variables=2)
        solutions = [problem.create_solution() for _ in range(3)]
        return AlgorithmState(
            problem=problem, evaluations=42, solutions=solutions, computing_time=1.5
        )

    def test_as_dict_uses_the_legacy_string_keys(self):
        state = self._state()

        as_dict = state.as_dict()

        assert set(as_dict.keys()) == {"PROBLEM", "EVALUATIONS", "SOLUTIONS", "COMPUTING_TIME"}

    def test_as_dict_values_match_the_typed_fields(self):
        state = self._state()

        as_dict = state.as_dict()

        assert as_dict["PROBLEM"] is state.problem
        assert as_dict["EVALUATIONS"] == 42
        assert as_dict["SOLUTIONS"] is state.solutions
        assert as_dict["COMPUTING_TIME"] == 1.5

    def test_as_dict_is_usable_as_observer_kwargs(self):
        received = {}

        def fake_observer_update(*args, **kwargs):
            received.update(kwargs)

        fake_observer_update(**self._state().as_dict())

        assert received["EVALUATIONS"] == 42
