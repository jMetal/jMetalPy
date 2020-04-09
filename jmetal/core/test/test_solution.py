import copy
import unittest

from jmetal.core.solution import BinarySolution, FloatSolution, IntegerSolution, Solution, CompositeSolution
from jmetal.util.ckecking import InvalidConditionException


class SolutionTestCase(unittest.TestCase):

    def test_should_default_constructor_create_a_valid_solution(self) -> None:
        solution = Solution(2, 3)
        self.assertEqual(2, solution.number_of_variables)
        self.assertEqual(3, solution.number_of_objectives)
        self.assertEqual(0, len(solution.attributes))
        self.assertEqual(2, len(solution.variables))
        self.assertEqual(3, len(solution.objectives))


class BinarySolutionTestCase(unittest.TestCase):
    def test_should_default_constructor_create_a_valid_solution(self) -> None:
        solution = BinarySolution(2, 3)
        self.assertEqual(2, solution.number_of_variables)
        self.assertEqual(3, solution.number_of_objectives)

    def test_should_constructor_create_a_valid_solution(self) -> None:
        solution = BinarySolution(number_of_variables=2, number_of_objectives=3)
        solution.variables[0] = [True, False]
        solution.variables[1] = [False]
        self.assertEqual(2, solution.number_of_variables)
        self.assertEqual(3, solution.number_of_objectives)
        self.assertEqual(2, len(solution.variables))
        self.assertEqual(3, len(solution.objectives))
        self.assertEqual([True, False], solution.variables[0])
        self.assertEqual([False], solution.variables[1])

    def test_should_get_total_number_of_bits_return_zero_if_the_object_variables_are_not_initialized(self) -> None:
        solution = BinarySolution(number_of_variables=2, number_of_objectives=3)
        self.assertEqual(0, solution.get_total_number_of_bits())

    def test_should_get_total_number_of_bits_return_the_right_value(self) -> None:
        solution = BinarySolution(number_of_variables=2, number_of_objectives=3)
        solution.variables[0] = [True, False]
        solution.variables[1] = [False, True, False]
        self.assertEqual(5, solution.get_total_number_of_bits())


class FloatSolutionTestCase(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self) -> None:
        solution = FloatSolution([], [], 2)
        self.assertIsNotNone(solution)

    def test_should_default_constructor_create_a_valid_solution(self) -> None:
        solution = FloatSolution([0.0, 0.5], [1.0, 2.0], 3)

        self.assertEqual(2, solution.number_of_variables)
        self.assertEqual(3, solution.number_of_objectives)
        self.assertEqual(2, len(solution.variables))
        self.assertEqual(3, len(solution.objectives))
        self.assertEqual([0.0, 0.5], solution.lower_bound)
        self.assertEqual([1.0, 2.0], solution.upper_bound)

    def test_should_copy_work_properly(self) -> None:
        solution = FloatSolution([0.0, 3.5], [1.0, 5.0], 3, 2)
        solution.variables = [1.24, 2.66]
        solution.objectives = [0.16, -2.34, 9.25]
        solution.constraints = [-1.2, -0.25]
        solution.attributes["attr"] = "value"

        new_solution = copy.copy(solution)

        self.assertEqual(solution.number_of_variables, new_solution.number_of_variables)
        self.assertEqual(solution.number_of_objectives, new_solution.number_of_objectives)
        self.assertEqual(solution.variables, new_solution.variables)
        self.assertEqual(solution.objectives, new_solution.objectives)
        self.assertEqual(solution.lower_bound, new_solution.lower_bound)
        self.assertEqual(solution.upper_bound, new_solution.upper_bound)
        self.assertEqual(solution.constraints, new_solution.constraints)
        self.assertIs(solution.lower_bound, solution.lower_bound)
        self.assertIs(solution.upper_bound, solution.upper_bound)
        self.assertEqual(solution.attributes, new_solution.attributes)


class IntegerSolutionTestCase(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self) -> None:
        solution = IntegerSolution([], [], 2)
        self.assertIsNotNone(solution)

    def test_should_default_constructor_create_a_valid_solution(self) -> None:
        solution = IntegerSolution([0, 5], [1, 2], 3, 0)

        self.assertEqual(2, solution.number_of_variables)
        self.assertEqual(3, solution.number_of_objectives)
        self.assertEqual(2, len(solution.variables))
        self.assertEqual(3, len(solution.objectives))
        self.assertEqual(0, len(solution.constraints))
        self.assertEqual([0, 5], solution.lower_bound)
        self.assertEqual([1, 2], solution.upper_bound)

    def test_should_copy_work_properly(self) -> None:
        solution = IntegerSolution([0, 5], [1, 2], 3, 1)
        solution.variables = [1, 2]
        solution.objectives = [0.16, -2.34, 9.25]
        solution.constraints = [-5]
        solution.attributes["attr"] = "value"

        new_solution = copy.copy(solution)

        self.assertEqual(solution.number_of_variables, new_solution.number_of_variables)
        self.assertEqual(solution.number_of_objectives, new_solution.number_of_objectives)
        self.assertEqual(solution.variables, new_solution.variables)
        self.assertEqual(solution.objectives, new_solution.objectives)
        self.assertEqual(solution.lower_bound, new_solution.lower_bound)
        self.assertEqual(solution.upper_bound, new_solution.upper_bound)
        self.assertEqual(solution.constraints, new_solution.constraints)
        self.assertIs(solution.lower_bound, solution.lower_bound)
        self.assertIs(solution.upper_bound, solution.upper_bound)
        self.assertEqual(solution.attributes, new_solution.attributes)


class CompositeSolutionTestCase(unittest.TestCase):
    def test_should_constructor_create_a_valid_not_none_composite_solution_composed_of_a_double_solution(self):
        composite_solution = CompositeSolution([FloatSolution([1.0], [2.0], 2)])
        self.assertIsNotNone(composite_solution)

    def test_should_constructor_raise_an_exception_if_the_number_of_objectives_is_not_coherent(self):
        float_solution: FloatSolution = FloatSolution([1.0], [3.0], 3)
        integer_solution: IntegerSolution = IntegerSolution([2], [4], 2)

        with self.assertRaises(InvalidConditionException):
            CompositeSolution([float_solution, integer_solution])

    def test_should_constructor_create_a_valid_soltion_composed_of_a_float_and_an_integer_solutions(self):
        number_of_objectives = 3
        number_of_constraints = 1
        float_solution: FloatSolution = FloatSolution([1.0], [3.0], number_of_objectives, number_of_constraints)
        integer_solution: IntegerSolution = IntegerSolution([2], [4], number_of_objectives, number_of_constraints)

        solution: CompositeSolution = CompositeSolution([float_solution, integer_solution])

        self.assertIsNotNone(solution)
        self.assertEqual(2, solution.number_of_variables)
        self.assertEqual(number_of_objectives, solution.number_of_objectives)
        self.assertEqual(number_of_constraints, solution.number_of_constraints)
        self.assertEqual(number_of_objectives, solution.variables[0].number_of_objectives)
        self.assertEqual(number_of_objectives, solution.variables[1].number_of_objectives)
        self.assertEqual(number_of_constraints, solution.variables[0].number_of_constraints)
        self.assertEqual(number_of_constraints, solution.variables[1].number_of_constraints)
        self.assertTrue(type(solution.variables[0] is FloatSolution))
        self.assertTrue(type(solution.variables[1] is IntegerSolution))

    def test_should_copy_work_properly(self):
        number_of_objectives = 3
        number_of_constraints = 1
        float_solution: FloatSolution = FloatSolution([1.0], [3.0], number_of_objectives, number_of_constraints)
        integer_solution: IntegerSolution = IntegerSolution([2], [4], number_of_objectives, number_of_constraints)

        solution: CompositeSolution = CompositeSolution([float_solution, integer_solution])
        new_solution: CompositeSolution = copy.deepcopy(solution)

        self.assertEqual(solution.number_of_variables, new_solution.number_of_variables)
        self.assertEqual(solution.number_of_objectives, new_solution.number_of_objectives)
        self.assertEqual(solution.number_of_constraints, new_solution.number_of_constraints)

        self.assertEqual(solution.variables[0].number_of_variables, new_solution.variables[0].number_of_variables)
        self.assertEqual(solution.variables[1].number_of_variables, new_solution.variables[1].number_of_variables)
        self.assertEqual(solution.variables[0], new_solution.variables[0])
        self.assertEqual(solution.variables[1], new_solution.variables[1])

        self.assertEqual(solution.variables[0].variables, new_solution.variables[0].variables)
        self.assertEqual(solution.variables[1].variables, new_solution.variables[1].variables)


if __name__ == '__main__':
    unittest.main()
