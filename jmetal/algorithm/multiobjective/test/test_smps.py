import unittest
import numpy

from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.component.archive import BoundedArchive
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
from jmetal.operator.mutation import Polynomial


class SMPSOTestCases(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_valid_object(self):
        problem = self.__DummyFloatProblem()
        algorithm = SMPSO(
            problem=problem,
            swarm_size=100,
            max_evaluations=200,
            mutation=Polynomial(probability=1.0/problem.number_of_variables),
            leaders=BoundedArchive[FloatSolution](100)
        )

        self.assertEqual(1.5, algorithm.c1_min)
        self.assertEqual(2.5, algorithm.c1_max)
        self.assertEqual(1.5, algorithm.c2_min)
        self.assertEqual(2.5, algorithm.c2_max)
        self.assertEqual(0.1, algorithm.min_weight)
        self.assertEqual(0.1, algorithm.max_weight)
        self.assertEqual(-1.0, algorithm.change_velocity1)
        self.assertEqual(-1.0, algorithm.change_velocity2)
        self.assertEqual(200, algorithm.max_evaluations)
        self.assertEqual(100, len(algorithm.speed))

        numpy.testing.assert_array_almost_equal(numpy.array([2.0, 2.0]), algorithm.delta_max)
        numpy.testing.assert_array_almost_equal(algorithm.delta_max * -1.0, algorithm.delta_min)

    class __DummyFloatProblem(Problem[FloatSolution]):
        def __init__(self):
            self.number_of_variables = 2
            self.number_of_objectives = 2
            self.number_of_constraints = 0

            self.lower_bound = [-2.0 for i in range(self.number_of_variables)]
            self.upper_bound = [2.0 for i in range(self.number_of_variables)]

            FloatSolution.lower_bound = self.lower_bound
            FloatSolution.upper_bound = self.upper_bound


if __name__ == '__main__':
    unittest.main()

