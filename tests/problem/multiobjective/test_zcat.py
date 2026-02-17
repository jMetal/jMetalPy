import math
import unittest

from jmetal.problem import ZCAT1, ZCAT10, ZCAT20
from jmetal.problem.multiobjective.zcat import ZCAT14


class ZCAT1TestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = ZCAT1()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = ZCAT1()
        self.assertEqual(30, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(-0.5, problem.lower_bound[0])
        self.assertEqual(0.5, problem.upper_bound[0])
        self.assertEqual(-15.0, problem.lower_bound[-1])
        self.assertEqual(15.0, problem.upper_bound[-1])

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = ZCAT1()
        solution = problem.create_solution()

        self.assertEqual(30, len(solution.variables))
        self.assertEqual(2, len(solution.objectives))
        self.assertTrue(all(problem.lower_bound[i] <= value for i, value in enumerate(solution.variables)))
        self.assertTrue(all(value <= problem.upper_bound[i] for i, value in enumerate(solution.variables)))

    def test_should_get_name_return_the_right_name(self) -> None:
        problem = ZCAT1()
        self.assertEqual("ZCAT1", problem.name())

    def test_should_evaluate_return_valid_objectives(self) -> None:
        problem = ZCAT1()
        solution = problem.create_solution()
        solution.variables = list(problem.lower_bound)

        problem.evaluate(solution)

        self.assertEqual(2, len(solution.objectives))
        self.assertTrue(all(math.isfinite(value) for value in solution.objectives))


class ZCATOptionsTestCases(unittest.TestCase):
    def test_should_allow_problem_with_one_decision_variable_when_pareto_set_is_one_dimensional(self) -> None:
        problem = ZCAT14(number_of_variables=1, number_of_objectives=3)
        solution = problem.create_solution()
        solution.variables = [0.0]

        problem.evaluate(solution)

        self.assertEqual(3, len(solution.objectives))
        self.assertTrue(all(math.isfinite(value) for value in solution.objectives))

    def test_should_fail_when_number_of_variables_is_not_valid(self) -> None:
        with self.assertRaises(ValueError):
            ZCAT1(number_of_variables=1, number_of_objectives=3)

    def test_should_support_all_config_flags(self) -> None:
        problem = ZCAT10(
            number_of_variables=30,
            number_of_objectives=2,
            complicated_pareto_set=True,
            level=4,
            bias=True,
            imbalance=True,
        )
        solution = problem.create_solution()
        solution.variables = list(problem.upper_bound)

        problem.evaluate(solution)

        self.assertEqual(2, len(solution.objectives))
        self.assertTrue(all(math.isfinite(value) for value in solution.objectives))

    def test_should_get_name_return_the_right_name_for_last_instance(self) -> None:
        problem = ZCAT20()
        self.assertEqual("ZCAT20", problem.name())


if __name__ == "__main__":
    unittest.main()
