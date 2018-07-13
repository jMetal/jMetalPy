import unittest

from jmetal.core.solution import Solution
from jmetal.component.ranking import FastNonDominatedRanking


class FastNonDominatedRankingTestCases(unittest.TestCase):

    def setUp(self):
        self.ranking = FastNonDominatedRanking()

    def test_should_constructor_create_a_valid_object(self):
        self.assertIsNotNone(self.ranking)

    def test_should_compute_ranking_of_an_emtpy_solution_list_return_a_empty_list_of_subranks(self):
        solution_list = []

        self.assertEqual(0, len(self.ranking.compute_ranking(solution_list)))

    def test_should_compute_ranking_return_a_subfront_if_the_solution_list_contains_one_solution(self):
        solution = Solution(2, 3)
        solution_list = [solution]

        ranking = self.ranking.compute_ranking(solution_list)

        self.assertEqual(1, self.ranking.get_number_of_subfronts())
        self.assertEqual(solution, ranking[0][0])

    def test_should_compute_ranking_return_a_subfront_if_the_solution_list_contains_two_nondominated_solutions(self):
        solution = Solution(2, 2)
        solution.objectives = [1, 2]
        solution2 = Solution(2, 2)
        solution2.objectives = [2, 1]
        solution_list = [solution, solution2]

        ranking = self.ranking.compute_ranking(solution_list)

        self.assertEqual(1, self.ranking.get_number_of_subfronts())
        self.assertEqual(2, len(self.ranking.get_subfront(0)))
        self.assertEqual(solution, ranking[0][0])
        self.assertEqual(solution2, ranking[0][1])

    def test_should_compute_ranking_work_properly_case1(self):
        """ The list contains two solutions and one of them is dominated by the other one.
        """
        solution = Solution(2, 2)
        solution.objectives = [2, 3]
        solution2 = Solution(2, 2)
        solution2.objectives = [3, 6]
        solution_list = [solution, solution2]

        ranking = self.ranking.compute_ranking(solution_list)

        self.assertEqual(2, self.ranking.get_number_of_subfronts())
        self.assertEqual(1, len(self.ranking.get_subfront(0)))
        self.assertEqual(1, len(self.ranking.get_subfront(1)))
        self.assertEqual(solution, ranking[0][0])
        self.assertEqual(solution2, ranking[1][0])

    def test_should_ranking_of_a_population_with_three_dominated_solutions_return_three_subfronts(self):
        solution = Solution(2, 2)
        solution.objectives = [2, 3]
        solution2 = Solution(2, 2)
        solution2.objectives = [3, 6]
        solution3 = Solution(2, 2)
        solution3.objectives = [4, 8]
        solution_list = [solution, solution2, solution3]

        ranking = self.ranking.compute_ranking(solution_list)

        self.assertEqual(3, self.ranking.get_number_of_subfronts())
        self.assertEqual(1, len(self.ranking.get_subfront(0)))
        self.assertEqual(1, len(self.ranking.get_subfront(1)))
        self.assertEqual(1, len(self.ranking.get_subfront(2)))
        self.assertEqual(solution, ranking[0][0])
        self.assertEqual(solution2, ranking[1][0])
        self.assertEqual(solution3, ranking[2][0])

    def test_should_ranking_of_a_population_with_five_solutions_work_properly(self):
        solution1 = Solution(2, 2)
        solution2 = Solution(2, 2)
        solution3 = Solution(2, 2)
        solution4 = Solution(2, 2)
        solution5 = Solution(2, 2)

        solution1.objectives[0] = 1.0
        solution1.objectives[1] = 0.0
        solution2.objectives[0] = 0.5
        solution2.objectives[1] = 0.5
        solution3.objectives[0] = 0.0
        solution3.objectives[1] = 1.0

        solution4.objectives[0] = 0.6
        solution4.objectives[1] = 0.6
        solution5.objectives[0] = 0.7
        solution5.objectives[1] = 0.5

        solution_list = [solution1, solution2, solution3, solution4, solution5]

        ranking = self.ranking.compute_ranking(solution_list)

        self.assertEqual(2, self.ranking.get_number_of_subfronts())
        self.assertEqual(3, len(self.ranking.get_subfront(0)))
        self.assertEqual(2, len(self.ranking.get_subfront(1)))
        self.assertEqual(solution1, ranking[0][0])
        self.assertEqual(solution2, ranking[0][1])
        self.assertEqual(solution3, ranking[0][2])
        self.assertEqual(solution4, ranking[1][0])
        self.assertEqual(solution5, ranking[1][1])


if __name__ == "__main__":
    unittest.main()
