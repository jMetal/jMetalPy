import unittest

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.core.quality_indicator import HyperVolume
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem import ZDT1
from jmetal.util.archive import CrowdingDistanceArchive
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
            termination_criterion=StoppingByEvaluations(max_evaluations=1000)
        ).run()

    def test_SMPSO(self):
        SMPSO(
            problem=self.problem,
            swarm_size=self.population_size,
            mutation=self.mutation,
            leaders=CrowdingDistanceArchive(100),
            termination_criterion=StoppingByEvaluations(max_evaluations=1000)
        ).run()


class IntegrationTestCases(unittest.TestCase):

    def test_should_NSGAII_work_when_solving_problem_ZDT1_with_standard_settings(self):
        problem = ZDT1()

        max_evaluations = 25000

        algorithm = NSGAII(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            crossover=SBXCrossover(probability=1.0, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        )

        algorithm.run()
        front = algorithm.get_result()

        hv = HyperVolume(reference_point=[1, 1])
        value = hv.compute([front[i].objectives for i in range(len(front))])

        self.assertTrue(value >= 0.65)

    def test_should_SMPSO_work_when_solving_problem_ZDT1_with_standard_settings(self):
        problem = ZDT1()

        algorithm = SMPSO(
            problem=problem,
            swarm_size=100,
            mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            leaders=CrowdingDistanceArchive(100),
            termination_criterion=StoppingByEvaluations(max_evaluations=25000)
        )

        algorithm.run()
        front = algorithm.get_result()

        hv = HyperVolume(reference_point=[1, 1])
        value = hv.compute([front[i].objectives for i in range(len(front))])

        self.assertTrue(value >= 0.655)


if __name__ == '__main__':
    unittest.main()
