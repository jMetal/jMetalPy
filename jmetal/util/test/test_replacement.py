import unittest

from jmetal.core.solution import Solution
from jmetal.util.density_estimator import KNearestNeighborDensityEstimator
from jmetal.util.ranking import StrengthRanking, FastNonDominatedRanking
from jmetal.util.replacement import RankingAndDensityEstimatorReplacement


class RankingAndDensityEstimatorReplacementTestCases(unittest.TestCase):

    def test_should_replacement_return_the_list_if_the_offspring_list_is_empty(self):
        """
         5 1
         4   2
         3     3
         2
         1         4
         0 1 2 3 4 5
        """
        ranking = StrengthRanking()
        density_estimator = KNearestNeighborDensityEstimator(1)

        replacement = RankingAndDensityEstimatorReplacement(ranking, density_estimator)

        solution1 = Solution(2, 2)
        solution1.objectives = [1, 5]
        solution2 = Solution(2, 2)
        solution2.objectives = [2, 4]
        solution3 = Solution(2, 2)
        solution3.objectives = [3, 3]
        solution4 = Solution(2, 2)
        solution4.objectives = [5, 1]

        solution_list = [solution1, solution2, solution3, solution4]
        result_list = replacement.replace(solution_list, [])

        self.assertEqual(4, len(result_list))
        self.assertEqual(0, solution1.attributes['strength_ranking'])
        self.assertEqual(0, solution2.attributes['strength_ranking'])
        self.assertEqual(0, solution3.attributes['strength_ranking'])
        self.assertEqual(0, solution4.attributes['strength_ranking'])

    def test_should_replacement_return_the_right_value_case1(self):
        """
         5 1
         4   2
         3     3
         2
         1         4
         0 1 2 3 4 5

         List: 1,2,3   OffspringList: 4
         Expected result: 4, 1, 3
        """
        ranking = StrengthRanking()
        density_estimator = KNearestNeighborDensityEstimator(1)

        replacement = RankingAndDensityEstimatorReplacement(ranking, density_estimator)

        solution1 = Solution(2, 2)
        solution1.objectives = [1, 5]
        solution2 = Solution(2, 2)
        solution2.objectives = [2, 4]
        solution3 = Solution(2, 2)
        solution3.objectives = [3, 3]
        solution4 = Solution(2, 2)
        solution4.objectives = [5, 1]

        solution_list = [solution1, solution2, solution3]
        offspring_list = [solution4]
        result_list = replacement.replace(solution_list, offspring_list)

        self.assertEqual(3, len(result_list))
        self.assertTrue(solution1 in result_list)
        self.assertTrue(solution3 in result_list)
        self.assertTrue(solution4 in result_list)

    def test_should_replacement_return_the_right_value_case2(self):
        """
         5 1
         4   2
         3     3
         2    5
         1         4
         0 1 2 3 4 5

         List: 1,2,4   OffspringList: 3,5
         Expected result: 1, 5, 4
        """
        ranking = StrengthRanking()
        density_estimator = KNearestNeighborDensityEstimator(1)

        replacement = RankingAndDensityEstimatorReplacement(ranking, density_estimator)

        solution1 = Solution(2, 2)
        solution1.objectives = [1, 5]
        solution2 = Solution(2, 2)
        solution2.objectives = [2, 4]
        solution3 = Solution(2, 2)
        solution3.objectives = [3, 3]
        solution4 = Solution(2, 2)
        solution4.objectives = [5, 1]
        solution5 = Solution(2, 2)
        solution5.objectives = [2.5, 2.5]

        solution_list = [solution1, solution2, solution4]
        offspring_list = [solution3, solution5]
        result_list = replacement.replace(solution_list, offspring_list)

        self.assertEqual(0, solution1.attributes['strength_ranking'])
        self.assertEqual(0, solution2.attributes['strength_ranking'])
        self.assertEqual(1, solution3.attributes['strength_ranking'])
        self.assertEqual(0, solution4.attributes['strength_ranking'])
        self.assertEqual(0, solution5.attributes['strength_ranking'])

        self.assertEqual(3, len(result_list))
        self.assertTrue(solution1 in result_list)
        self.assertTrue(solution5 in result_list)
        self.assertTrue(solution4 in result_list)

    def test_should_replacement_return_the_right_value_case3(self):
        """
         """

        points_population = [[0.13436424411240122, 4.323216008886963],
                             [0.23308445025757263, 4.574937990387161],
                             [0.17300740157905092, 4.82329350808847],
                             [0.9571162814602269, 3.443495331489301],
                             [0.25529404008730594, 3.36387501100745],
                             [0.020818108509287336, 5.1051826661880515],
                             [0.8787178982088466, 3.2716009445324103],
                             [0.6744550697237632, 3.901350307095427],
                             [0.7881164487252263, 3.1796004913916516],
                             [0.1028341459863098, 4.9409270526888935]]

        points_offspring_population = [[0.3150521745650882, 4.369120371847888],
                                       [0.8967291504209932, 2.506948771242972],
                                       [0.6744550697237632, 3.9361442668874504],
                                       [0.9571162814602269, 3.4388386707431433],
                                       [0.13436424411240122, 4.741872175943253],
                                       [0.25529404008730594, 2.922302861104415],
                                       [0.23308445025757263, 4.580180404770213],
                                       [0.23308445025757263, 4.591260299892424],
                                       [0.9571162814602269, 2.9865495383518694],
                                       [0.25529404008730594, 3.875587748122183]]

        ranking = FastNonDominatedRanking()
        density_estimator = KNearestNeighborDensityEstimator(1)

        population = []
        for i in range(len(points_population)):
            population.append(Solution(2, 2))
            population[i].objectives = points_population[i]

        offspring_population = []
        for i in range(len(points_offspring_population)):
            offspring_population.append(Solution(2, 2))
            offspring_population[i].objectives = points_offspring_population[i]

        replacement = RankingAndDensityEstimatorReplacement(ranking, density_estimator)
        result_list = replacement.replace(population, offspring_population)

        self.assertEqual(10,len(result_list))

        for solution in result_list[0:4]:
            self.assertEqual(0, solution.attributes['dominance_ranking'])
        for solution in result_list[5:9]:
            self.assertEqual(1, solution.attributes['dominance_ranking'])




if __name__ == '__main__':
    unittest.main()
