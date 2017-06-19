import unittest

from jmetal.core.solution import Solution
from jmetal.util.ranking import DominanceRanking


class DominanceRankingTestCases(unittest.TestCase):
    def setUp(self):
        self.ranking = DominanceRanking()

    def test_should_constructor_create_a_valid_object(self):
        self.assertIsNotNone(self.ranking)

    def test_should_compute_ranking_of_an_emtpy_solution_list_return_a_empty_list_of_subranks(self):
        solution_list = []

        self.assertEquals(0, len(self.ranking.compute_ranking(solution_list)))

    def test_should_compute_ranking_return_a_subfront_if_the_solution_list_contains_one_solution(self):
        solution = Solution(2, 3)
        solution_list = [solution]

        ranking = self.ranking.compute_ranking(solution_list)

        self.assertEquals(1, len(ranking))
        self.assertEquals(solution, solution_list[ranking[0][0]])

    def test_should_compute_ranking_return_a_subfront_if_the_solution_list_contains_two_nondominated_solutions(self):
        solution = Solution(2, 2)
        solution.objectives = [1, 2]
        solution2 = Solution(2, 2)
        solution2.objectives = [2, 1]
        solution_list = [solution, solution2]

        ranking = self.ranking.compute_ranking(solution_list)

        self.assertEquals(1, len(ranking))
        self.assertEquals(2, len(ranking[0]))
        self.assertEquals(solution, solution_list[ranking[0][0]])
        self.assertEquals(solution2, solution_list[ranking[0][1]])

    def test_should_compute_ranking_work_properly_case1(self):
        """ The list contains two solutions and one of them is dominated by the other one """
        solution = Solution(2, 2)
        solution.objectives = [1, 2]
        solution2 = Solution(2, 2)
        solution2.objectives = [0, 1]
        solution_list = [solution, solution2]

        ranking = self.ranking.compute_ranking(solution_list)

        self.assertEquals(2, len(ranking))
        self.assertEquals(1, len(ranking[0]))
        self.assertEquals(1, len(ranking[1]))
        self.assertEquals(solution, solution_list[ranking[0][0]])
        self.assertEquals(solution2, solution_list[ranking[1][0]])

if __name__ == "__main__":
    unittest.main()