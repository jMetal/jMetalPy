import unittest

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.smpso import SMPSO
<<<<<<< HEAD
from jmetal.operator import PolynomialMutation, SBXCrossover, BinaryTournamentSelection
from jmetal.problem import ZDT1
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.comparator import RankingAndCrowdingDistanceComparator
=======
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem import ZDT1
from jmetal.util.archive import CrowdingDistanceArchive
>>>>>>> 52e0b172f0c6d651ba08b961a90a382f0a4b8e0f
from jmetal.util.termination_criterion import StoppingByEvaluations


class RunningAlgorithmsTestCases(unittest.TestCase):

    def setUp(self):
        self.problem = ZDT1()
        self.population_size = 100
        self.offspring_size = 100
        self.mating_pool_size = 100
        self.max_evaluations = 100
        self.mutation = PolynomialMutation(probability=1.0 / self.problem.number_of_variables, distribution_index=20)
        self.crossover = SBXCrossover(probability=1.0, distribution_index=20)

    def test_NSGAII(self):
        NSGAII(
            problem=self.problem,
            population_size=self.population_size,
            offspring_population_size=self.offspring_size,
            mutation=self.mutation,
            crossover=self.crossover,
<<<<<<< HEAD
            selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
=======
>>>>>>> 52e0b172f0c6d651ba08b961a90a382f0a4b8e0f
            termination_criterion=StoppingByEvaluations(max=1000)
        ).run()

    def test_SMPSO(self):
        SMPSO(
            problem=self.problem,
            swarm_size=self.population_size,
            mutation=self.mutation,
            leaders=CrowdingDistanceArchive(100),
            termination_criterion=StoppingByEvaluations(max=1000)
        ).run()


if __name__ == '__main__':
    unittest.main()
