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

        self.assertEquals(1, len(self.ranking.compute_ranking(solution_list)))


if __name__ == "__main__":
    unittest.main()