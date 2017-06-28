import unittest

from jmetal.algorithm.multiobjective.nsgaii import RankingAndCrowdingDistanceSelection
from jmetal.core.solution import Solution
from jmetal.util.ranking import DominanceRanking


class DominanceRankingTestCases(unittest.TestCase):
    def setUp(self):
        self.ranking_and_crowding_selection = RankingAndCrowdingDistanceSelection(5)

    def test_should_len_of_nsgaii_execute_be_5(self):
        solution1 = Solution(2, 2)
        solution2 = Solution(2, 2)
        solution3 = Solution(2, 2)
        solution4 = Solution(2, 2)
        solution5 = Solution(2, 2)
        solution6 = Solution(2, 2)

        solution1.objectives[0] = 1.0
        solution1.objectives[1] = 0.0
        solution2.objectives[0] = 0.6
        solution2.objectives[1] = 0.6
        solution3.objectives[0] = 0.5
        solution3.objectives[1] = 0.5
        solution4.objectives[0] = 1.1
        solution4.objectives[1] = 0.0
        solution5.objectives[0] = 0.0
        solution5.objectives[1] = 1.0
        solution6.objectives[0] = 1.05
        solution6.objectives[1] = 0.1

        solution_list = [solution1, solution2, solution3, solution4, solution5, solution6]

        list_of_crowding_and_rankings = self.ranking_and_crowding_selection.execute(solution_list)

        self.assertEqual(len(list_of_crowding_and_rankings), 5)
        self.assertEqual(solution1, list_of_crowding_and_rankings[0])
        self.assertEqual(solution3, list_of_crowding_and_rankings[1])
        self.assertEqual(solution5, list_of_crowding_and_rankings[2])
        self.assertEqual(solution4, list_of_crowding_and_rankings[3])
        self.assertEqual(solution2, list_of_crowding_and_rankings[4])

if __name__ == "__main__":
    unittest.main()
