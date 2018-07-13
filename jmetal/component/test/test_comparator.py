import unittest

from jmetal.core.solution import FloatSolution, Solution
from jmetal.component.comparator import DominanceComparator, SolutionAttributeComparator, \
    RankingAndCrowdingDistanceComparator


class DominanceComparatorTestCases(unittest.TestCase):

    def setUp(self):
        self.comparator = DominanceComparator()

    def test_should_dominance_comparator_raise_an_exception_if_the_first_solution_is_null(self):
        solution = None
        solution2 = FloatSolution(3, 2, 0, [], [])
        with self.assertRaises(Exception):
            self.comparator.compare(solution, solution2)

    def test_should_dominance_comparator_raise_an_exception_if_the_second_solution_is_null(self):
        solution = FloatSolution(3, 2, 0, [], [])
        solution2 = None
        with self.assertRaises(Exception):
            self.comparator.compare(solution, solution2)

    def test_should_dominance_comparator_return_zero_if_the_two_solutions_have_one_objective_with_the_same_value(self):
        solution = FloatSolution(3, 1, 0, [], [])
        solution2 = FloatSolution(3, 1, 0, [], [])

        solution.objectives = [1.0]
        solution2.objectives = [1.0]

        self.assertEqual(0, self.comparator.compare(solution, solution2))

    def test_should_dominance_comparator_return_one_if_the_two_solutions_have_one_objective_and_the_second_one_is_lower(self):
        solution = FloatSolution(3, 1, 0, [], [])
        solution2 = FloatSolution(3, 1, 0, [], [])

        solution.objectives = [2.0]
        solution2.objectives = [1.0]

        self.assertEqual(1, self.comparator.compare(solution, solution2))

    def test_should_dominance_comparator_return_minus_one_if_the_two_solutions_have_one_objective_and_the_first_one_is_lower(self):
        solution = FloatSolution(3, 1, 0, [], [])
        solution2 = FloatSolution(3, 1, 0, [], [])

        solution.objectives = [1.0]
        solution2.objectives = [2.0]

        self.assertEqual(-1, self.comparator.compare(solution, solution2))

    def test_should_dominance_comparator_work_properly_case_a(self):
        """ Case A: solution1 has objectives [-1.0, 5.0, 9.0] and solution2 has [2.0, 6.0, 15.0]
        """
        solution = FloatSolution(3, 3, 0, [], [])
        solution2 = FloatSolution(3, 3, 0, [], [])

        solution.objectives = [-1.0, 5.0, 9.0]
        solution2.objectives = [2.0, 6.0, 15.0]

        self.assertEqual(-1, self.comparator.compare(solution, solution2))

    def test_should_dominance_comparator_work_properly_case_b(self):
        """ Case b: solution1 has objectives [-1.0, 5.0, 9.0] and solution2 has [-1.0, 5.0, 10.0]
        """
        solution = FloatSolution(3, 3, 0, [], [])
        solution2 = FloatSolution(3, 3, 0, [], [])

        solution.objectives = [-1.0, 5.0, 9.0]
        solution2.objectives = [-1.0, 5.0, 10.0]

        self.assertEqual(-1, self.comparator.compare(solution, solution2))

    def test_should_dominance_comparator_work_properly_case_c(self):
        """ Case c: solution1 has objectives [-1.0, 5.0, 9.0] and solution2 has [-2.0, 5.0, 9.0]
        """
        solution = FloatSolution(3, 3, 0, [], [])
        solution2 = FloatSolution(3, 3, 0, [], [])

        solution.objectives = [-1.0, 5.0, 9.0]
        solution2.objectives = [-2.0, 5.0, 9.0]

        self.assertEqual(1, self.comparator.compare(solution, solution2))

    def test_should_dominance_comparator_work_properly_case_d(self):
        """ Case d: solution1 has objectives [-1.0, 5.0, 9.0] and solution2 has [-1.0, 5.0, 8.0]
        """
        solution = FloatSolution(3, 3, 0, [], [])
        solution2 = FloatSolution(3, 3, 0, [], [])

        solution.objectives = [-1.0, 5.0, 9.0]
        solution2.objectives = [-1.0, 5.0, 8.0]

        self.assertEqual(1, self.comparator.compare(solution, solution2))

    def test_should_dominance_comparator_work_properly_case_3(self):
        """ Case d: solution1 has objectives [-1.0, 5.0, 9.0] and solution2 has [-2.0, 5.0, 10.0]
        """
        solution = FloatSolution(3, 3, 0, [], [])
        solution2 = FloatSolution(3, 3, 0, [], [])

        solution.objectives = [-1.0, 5.0, 9.0]
        solution2.objectives = [-2.0, 5.0, 10.0]

        self.assertEqual(0, self.comparator.compare(solution, solution2))

    def test_should_dominance_comparator_work_properly_with_constrains_case_1(self):
        """ Case 1: solution1 has a higher degree of constraint violation than solution 2
        """
        solution1 = FloatSolution(3, 3, 0, [], [])
        solution2 = FloatSolution(3, 3, 0, [], [])
        solution1.attributes["overall_constraint_violation"] = -0.1
        solution2.attributes["overall_constraint_violation"] = -0.3

        solution1.objectives = [-1.0, 5.0, 9.0]
        solution2.objectives = [-2.0, 5.0, 10.0]

        self.assertEqual(-1, self.comparator.compare(solution1, solution2))

    def test_should_dominance_comparator_work_properly_with_constrains_case_2(self):
        """ Case 2: solution1 has a lower degree of constraint violation than solution 2
        """
        solution1 = FloatSolution(3, 3, 0, [], [])
        solution2 = FloatSolution(3, 3, 0, [], [])
        solution1.attributes["overall_constraint_violation"] = -0.3
        solution2.attributes["overall_constraint_violation"] = -0.1

        solution1.objectives = [-1.0, 5.0, 9.0]
        solution2.objectives = [-2.0, 5.0, 10.0]

        self.assertEqual(1, self.comparator.compare(solution1, solution2))


class SolutionAttributeComparatorTestCases(unittest.TestCase):

    def setUp(self):
        self.comparator = SolutionAttributeComparator("attribute")

    def test_should_compare_return_zero_if_the_first_solution_has_no_the_attribute(self):
        solution1 = Solution(1, 1)
        solution2 = Solution(1, 1)
        solution2.attributes["attribute"] = 1.0

        self.assertEqual(0, self.comparator.compare(solution1, solution2))

    def test_should_compare_return_zero_if_the_second_solution_has_no_the_attribute(self):
        solution1 = Solution(1, 1)
        solution2 = Solution(1, 1)
        solution1.attributes["attribute"] = 1.0

        self.assertEqual(0, self.comparator.compare(solution1, solution2))

    def test_should_compare_return_zero_if_none_of_the_solutions_have_the_attribute(self):
        solution1 = Solution(1, 1)
        solution2 = Solution(1, 1)

        self.assertEqual(0, self.comparator.compare(solution1, solution2))

    def test_should_compare_return_zero_if_both_solutions_have_the_same_attribute_value(self):
        solution1 = Solution(1, 1)
        solution2 = Solution(1, 1)
        solution1.attributes["attribute"] = 1.0
        solution2.attributes["attribute"] = 1.0

        self.assertEqual(0, self.comparator.compare(solution1, solution2))

    def test_should_compare_works_properly_case1(self):
        """ Case 1: solution1.attribute < solution2.attribute (lowest is best)
        """
        solution1 = Solution(1, 1)
        solution2 = Solution(1, 1)
        solution1.attributes["attribute"] = 0.0
        solution2.attributes["attribute"] = 1.0

        self.assertEqual(-1, self.comparator.compare(solution1, solution2))

    def test_should_compare_works_properly_case2(self):
        """ Case 2: solution1.attribute > solution2.attribute (lowest is best)
        """
        solution1 = Solution(1, 1)
        solution2 = Solution(1, 1)
        solution1.attributes["attribute"] = 1.0
        solution2.attributes["attribute"] = 0.0

        self.assertEqual(1, self.comparator.compare(solution1, solution2))

    def test_should_compare_works_properly_case3(self):
        """ Case 3: solution1.attribute < solution2.attribute (highest is best)
        """
        comparator = SolutionAttributeComparator("attribute", False)
        solution1 = Solution(1, 1)
        solution2 = Solution(1, 1)
        solution1.attributes["attribute"] = 0.0
        solution2.attributes["attribute"] = 1.0

        self.assertEqual(1, comparator.compare(solution1, solution2))

    def test_should_compare_works_properly_case4(self):
        """ Case 4: solution1.attribute > solution2.attribute (highest is best)
        """
        solution1 = Solution(1, 1)
        solution2 = Solution(1, 1)
        solution1.attributes["attribute"] = 1.0
        solution2.attributes["attribute"] = 0.0

        comparator = SolutionAttributeComparator("attribute", False)
        self.assertEqual(-1, comparator.compare(solution1, solution2))


class RankingAndCrowdingComparatorTestCases(unittest.TestCase):

    def setUp(self):
        self.comparator = RankingAndCrowdingDistanceComparator()

    def test_should_compare_work_properly_case_1(self):
        """ Case 1: solution1.ranking < solution2.ranking
        """
        solution1 = Solution(1, 1)
        solution2 = Solution(1, 1)
        solution1.attributes["dominance_ranking"] = 1.0
        solution2.attributes["dominance_ranking"] = 2.0

        self.assertEqual(-1, self.comparator.compare(solution1, solution2))

    def test_should_compare_work_properly_case_2(self):
        """ Case 2: solution1.ranking > solution2.ranking
        """
        solution1 = Solution(1, 1)
        solution2 = Solution(1, 1)
        solution1.attributes["dominance_ranking"] = 2.0
        solution2.attributes["dominance_ranking"] = 1.0

        self.assertEqual(1, self.comparator.compare(solution1, solution2))

    def test_should_compare_work_properly_case_3(self):
        """ Case 3: solution1.ranking == solution2.ranking
                    solution1.crowding < solution2.crowding
        """
        solution1 = Solution(1, 1)
        solution2 = Solution(1, 1)
        solution1.attributes["dominance_ranking"] = 1.0
        solution2.attributes["dominance_ranking"] = 1.0
        solution1.attributes["crowding_distance"] = 1.0
        solution2.attributes["crowding_distance"] = 2.0

        self.assertEqual(1, self.comparator.compare(solution1, solution2))

    def test_should_compare_work_properly_case_4(self):
        """ Case 4: solution1.ranking == solution2.ranking
                    solution1.crowding > solution2.crowding
        """
        solution1 = Solution(1, 1)
        solution2 = Solution(1, 1)
        solution1.attributes["dominance_ranking"] = 1.0
        solution2.attributes["dominance_ranking"] = 1.0
        solution1.attributes["crowding_distance"] = 2.0
        solution2.attributes["crowding_distance"] = 1.0

        self.assertEqual(-1, self.comparator.compare(solution1, solution2))

    def test_should_compare_work_properly_case_5(self):
        """ Case 5: solution1.ranking == solution2.ranking
                    solution1.crowding == solution2.crowding
        """
        solution1 = Solution(1, 1)
        solution2 = Solution(1, 1)
        solution1.attributes["dominance_ranking"] = 1.0
        solution2.attributes["dominance_ranking"] = 1.0
        solution1.attributes["crowding_distance"] = 2.0
        solution2.attributes["crowding_distance"] = 2.0

        self.assertEqual(0, self.comparator.compare(solution1, solution2))


if __name__ == '__main__':
    unittest.main()
