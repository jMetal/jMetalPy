import unittest
from unittest import mock

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
        filename = 'resources/Knapsack_instances/KnapsackInstance_50_0_0.kp'

        data = "50\n13629\n 865 445\n395 324\n777 626\n912 656\n431 935\n42 210 \n266 990\n989 566\n524 489\n" \
               "498 454\n415 887\n941 534\n803 267\n850 64 \n311 825\n992 941\n489 562\n367 938\n598 15 \n914 96 \n" \
               "930 737\n224 861\n517 409\n143 728\n289 845\n144 804\n774 685\n98 641 \n634 2  \n819 627\n257 506\n" \
               "932 848\n546 889\n723 342\n830 250\n617 748\n924 334\n151 721\n318 892\n102 65 \n748 196\n76 940 \n" \
               "921 582\n871 228\n701 245\n339 823\n484 991\n574 146\n104 823\n363 557"
        with mock.patch('jmetal.problem.singleobjective.knapsack.open', new=mock.mock_open(read_data=data)):
            problem = Knapsack(from_file=True, filename=filename)
            self.assertEqual(1, problem.number_of_variables)
            self.assertEqual(1, problem.number_of_objectives)
            self.assertEqual(1, problem.number_of_constraints)
            self.assertEqual(50, problem.number_of_bits)

    def test_should_get_name_return_the_right_name(self):
        problem = Knapsack()
        self.assertEqual('Knapsack', problem.get_name())
