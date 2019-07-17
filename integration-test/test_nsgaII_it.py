import unittest
from typing import TypeVar

from jmetal.core.quality_indicator import HyperVolume

from jmetal.util.termination_criterion import StoppingByEvaluations

from jmetal.operator import PolynomialMutation, SBXCrossover

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.problem import ZDT1
from jmetal.util.solutions import read_solutions

S = TypeVar('S')
R = TypeVar('R')


class NSGAIIIntegrationTestCases(unittest.TestCase):

    def test_should_NSGAII_work_when_solving_problem_ZDT1_with_standard_settings(self) -> None:
        problem = ZDT1()

        max_evaluations = 25000
        algorithm = NSGAII(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            crossover=SBXCrossover(probability=1.0, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max=max_evaluations)
        )

        algorithm.run()
        front = algorithm.get_result()

        reference_point = [1, 1]
        hv = HyperVolume(reference_point)
        value = hv.compute(front)

        self.assertTrue(value >= 0.65)


if __name__ == '__main__':
    unittest.main()
