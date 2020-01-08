import unittest

from mockito import mock, when, verify, never

from jmetal.core.solution import Solution
from jmetal.util.comparator import DominanceComparator, SolutionAttributeComparator, \
    RankingAndCrowdingDistanceComparator, Comparator, OverallConstraintViolationComparator, MultiComparator


class OverallConstraintViolationComparatorTestCases(unittest.TestCase):
    def setUp(self):
        self.comparator: Comparator = OverallConstraintViolationComparator()

    def test_should_comparator_return_0_if_the_solutions_have_no_constraints(self):
        solution1 = Solution(1, 1, 0)
        solution2 = Solution(1, 1, 0)

        self.assertEqual(0, self.comparator.compare(solution1, solution2))

    def test_should_comparator_return_0_if_the_solutions_have_the_same_constraint_violation_degree(self):
        solution1 = Solution(1, 1, 2)
        solution2 = Solution(1, 1, 2)
        solution1.constraints[0] = -2
        solution1.constraints[1] = -3
        solution2.constraints[0] = -1
        solution2.constraints[1] = -4

        self.assertEqual(0, self.comparator.compare(solution1, solution2))

    def test_should_comparator_return_minus_1_if_solution_2_has_lower_constraint_violation_degree(self):
        solution1 = Solution(1, 1, 1)
        solution2 = Solution(1, 1, 1)
        solution1.constraints[0] = -2
        solution2.constraints[0] = -1

        self.assertEqual(1, self.comparator.compare(solution1, solution2))

    def test_should_comparator_return_1_if_solution_2_has_higher_constraint_violation_degree(self):
        solution1 = Solution(1, 1, 1)
        solution2 = Solution(1, 1, 1)
        solution1.constraints[0] = -2
        solution2.constraints[0] = -5

        self.assertEqual(-1, self.comparator.compare(solution1, solution2))


class DominanceComparatorTestCases(unittest.TestCase):

    def setUp(self):
        self.comparator = DominanceComparator()

    def test_should_dominance_comparator_raise_an_exception_if_the_first_solution_is_null(self):
        solution = None
        solution2 = Solution(2, 2)
        with self.assertRaises(Exception):
            self.comparator.compare(solution, solution2)

    def test_should_dominance_comparator_raise_an_exception_if_the_second_solution_is_null(self):
        solution = Solution(2, 3)
        solution2 = None
        with self.assertRaises(Exception):
            self.comparator.compare(solution, solution2)

    def test_should_dominance_comparator_return_zero_if_the_two_solutions_have_one_objective_with_the_same_value(self):
        solution = Solution(1, 1)
        solution2 = Solution(1, 1)
        solution.objectives = [1.0]
        solution2.objectives = [1.0]

        self.assertEqual(0, self.comparator.compare(solution, solution2))

    def test_should_dominance_comparator_return_one_if_the_two_solutions_have_one_objective_and_the_second_one_is_lower(
            self):
        solution = Solution(1, 1)
        solution2 = Solution(1, 1)
        solution.objectives = [2.0]
        solution2.objectives = [1.0]

        self.assertEqual(1, self.comparator.compare(solution, solution2))

    def test_should_dominance_comparator_return_minus_one_if_the_two_solutions_have_one_objective_and_the_first_one_is_lower(
            self):
        solution = Solution(1, 1)
        solution2 = Solution(1, 1)
        solution.objectives = [1.0]
        solution2.objectives = [2.0]

        self.assertEqual(-1, self.comparator.compare(solution, solution2))

    def test_should_dominance_comparator_work_properly_case_a(self):
        """ Case A: solution1 has objectives [-1.0, 5.0, 9.0] and solution2 has [2.0, 6.0, 15.0]
        """
        solution = Solution(1, 3)
        solution2 = Solution(1, 3)
        solution.objectives = [-1.0, 5.0, 9.0]
        solution2.objectives = [2.0, 6.0, 15.0]

        self.assertEqual(-1, self.comparator.compare(solution, solution2))

    def test_should_dominance_comparator_work_properly_case_b(self):
        """ Case b: solution1 has objectives [-1.0, 5.0, 9.0] and solution2 has [-1.0, 5.0, 10.0]
        """
        solution = Solution(1, 3)
        solution2 = Solution(1, 3)
        solution.objectives = [-1.0, 5.0, 9.0]
        solution2.objectives = [-1.0, 5.0, 10.0]

        self.assertEqual(-1, self.comparator.compare(solution, solution2))

    def test_should_dominance_comparator_work_properly_case_c(self):
        """ Case c: solution1 has objectives [-1.0, 5.0, 9.0] and solution2 has [-2.0, 5.0, 9.0]
        """
        solution = Solution(1, 3)
        solution2 = Solution(1, 3)
        solution.objectives = [-1.0, 5.0, 9.0]
        solution2.objectives = [-2.0, 5.0, 9.0]

        self.assertEqual(1, self.comparator.compare(solution, solution2))

    def test_should_dominance_comparator_work_properly_case_d(self):
        """ Case d: solution1 has objectives [-1.0, 5.0, 9.0] and solution2 has [-1.0, 5.0, 8.0]
        """
        solution = Solution(1, 3)
        solution2 = Solution(1, 3)
        solution.objectives = [-1.0, 5.0, 9.0]
        solution2.objectives = [-1.0, 5.0, 8.0]

        self.assertEqual(1, self.comparator.compare(solution, solution2))

    def test_should_dominance_comparator_work_properly_case_3(self):
        """ Case d: solution1 has objectives [-1.0, 5.0, 9.0] and solution2 has [-2.0, 5.0, 10.0]
        """
        solution = Solution(1, 3)
        solution2 = Solution(1, 3)
        solution.objectives = [-1.0, 5.0, 9.0]
        solution2.objectives = [-2.0, 5.0, 10.0]

        self.assertEqual(0, self.comparator.compare(solution, solution2))

    def test_should_dominance_comparator_work_properly_with_constrains_case_1(self):
        """ Case 1: solution1 has a higher degree of constraint violation than solution 2
        """
        solution1 = Solution(1, 3, 1)
        solution2 = Solution(1, 3, 1)

        solution1.constraints[0] = -0.1
        solution2.constraints[0] = -0.3

        solution1.objectives = [-1.0, 5.0, 9.0]
        solution2.objectives = [-2.0, 5.0, 10.0]

        self.assertEqual(-1, self.comparator.compare(solution1, solution2))

    def test_should_dominance_comparator_work_properly_with_constrains_case_2(self):
        """ Case 2: solution1 has a lower degree of constraint violation than solution 2
        """
        solution1 = Solution(1, 3, 1)
        solution2 = Solution(1, 3, 1)

        solution1.constraints[0] = -0.3
        solution2.constraints[0] = -0.1

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


class MultiComparatorTestCases(unittest.TestCase):

    def test_should_compare_return_zero_if_the_comparator_list_is_empty(self):
        solution1 = Solution(2, 2)
        solution2 = Solution(2, 2)

        multi_comparator = MultiComparator([])
        self.assertEqual(0, multi_comparator.compare(solution1, solution2))

    def test_should_compare_work_properly_case_1(self):
        """ Case 1: a comparator returning 0.
        """
        solution1 = Solution(2, 2)
        solution2 = Solution(2, 2)

        mocked_comparator: Comparator = mock()
        when(mocked_comparator).compare(solution1, solution2).thenReturn(0)

        comparator_list = [mocked_comparator]

        multi_comparator = MultiComparator(comparator_list)
        self.assertEqual(0, multi_comparator.compare(solution1, solution2))

        verify(mocked_comparator, times=1).compare(solution1, solution2)

    def test_should_compare_work_properly_case_2(self):
        """ Case 2: two comparators; the first returns 1 and the second one returns 0.
            Expected result: 1
        """
        solution1 = Solution(2, 2)
        solution2 = Solution(2, 2)

        mocked_comparator1: Comparator = mock()
        when(mocked_comparator1).compare(solution1, solution2).thenReturn(1)
        mocked_comparator2: Comparator = mock()
        when(mocked_comparator2).compare(solution1, solution2).thenReturn(0)

        comparator_list = [mocked_comparator1, mocked_comparator2]

        multi_comparator = MultiComparator(comparator_list)
        self.assertEqual(1, multi_comparator.compare(solution1, solution2))

        verify(mocked_comparator1, times=1).compare(solution1, solution2)
        verify(mocked_comparator2, never).compare(solution1, solution2)

    def test_should_compare_work_properly_case_3(self):
        """ Case 2: two comparators; the first returns 0 and the second one returns -1.
            Expected result: -1
        """
        solution1 = Solution(2, 2)
        solution2 = Solution(2, 2)

        mocked_comparator1: Comparator = mock()
        when(mocked_comparator1).compare(solution1, solution2).thenReturn(0)
        mocked_comparator2: Comparator = mock()
        when(mocked_comparator2).compare(solution1, solution2).thenReturn(-1)

        comparator_list = [mocked_comparator1, mocked_comparator2]

        multi_comparator = MultiComparator(comparator_list)
        self.assertEqual(-1, multi_comparator.compare(solution1, solution2))

        verify(mocked_comparator1, times=1).compare(solution1, solution2)
        verify(mocked_comparator2, times=1).compare(solution1, solution2)


if __name__ == '__main__':
    unittest.main()
