import unittest

from jmetal.problem.singleobjective.knapsack import Knapsack


class KnapsackTestCases(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = Knapsack()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = Knapsack()
        self.assertEqual(1, problem.number_of_variables)
        self.assertEqual(1, problem.number_of_objectives)
        self.assertEqual(1, problem.number_of_constraints)
        self.assertEqual(50, problem.number_of_bits)
        self.assertEqual(1000, problem.capacity)
        self.assertIsNone(problem.profits)
        self.assertIsNone(problem.weights)

    def test_should_constructor_create_a_valid_problem_with_500_bits(self) -> None:
        problem = Knapsack(500)
        self.assertEqual(1, problem.number_of_variables)
        self.assertEqual(1, problem.number_of_objectives)
        self.assertEqual(1, problem.number_of_constraints)
        self.assertEqual(500, problem.number_of_bits)
        self.assertEqual(1000, problem.capacity)
        self.assertIsNone(problem.profits)
        self.assertIsNone(problem.weights)

    def test_should_create_solution_a_valid_binary_solution(self) -> None:
        problem = Knapsack(256)
        solution = problem.create_solution()
        self.assertEqual(256, len(solution.variables[0]))

    def test_should_create_solution_from_file(self) -> None:
        filename = "resources/Knapsack_instances/KnapsackInstance_1000_7_93.kp"
        problem = Knapsack(from_file=True, filename=filename)
        self.assertEqual(1, problem.number_of_variables)
        self.assertEqual(1, problem.number_of_objectives)
        self.assertEqual(1, problem.number_of_constraints)
        self.assertEqual(1000, problem.number_of_bits)

    def test_should_get_name_return_the_right_name(self):
        problem = Knapsack()
        self.assertEqual("Knapsack", problem.get_name())
