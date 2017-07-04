from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.core.solution import BinarySolution, FloatSolution
from jmetal.operator.crossover import SinglePoint, SBX
from jmetal.operator.mutation import BitFlip, Polynomial
from jmetal.operator.selection import BinaryTournament
from jmetal.problem.singleobjectiveproblem import OneMax, Sphere

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import unittest
import pytest

class AlgorithmTestIntegrationTestCases(unittest.TestCase):

    def setUp(self):
        bits = 512
        self.problem = OneMax(bits)
        self.algorithm = GenerationalGeneticAlgorithm[BinarySolution, BinarySolution](
            self.problem,
            population_size=100,
            max_evaluations=25000,
            mutation=BitFlip(1.0 / bits),
            crossover=SinglePoint(0.9),
            selection=BinaryTournament())


    def test_genetic_algorithm(self):
        self.algorithm.run()
        result = self.algorithm.get_result()
        logger.info("Algorithm (binary problem): " + self.algorithm.get_name())
        logger.info("Problem: " +  self.problem.get_name())
        logger.info("Solution: " + str(result.variables[0]))
        logger.info("Fitness:  " + str(result.objectives[0]))
        logger.info("Computing time: " + str(self.algorithm.total_computing_time))

        assert(-500 <= result.objectives[0] <= -300 or 300 <= result.objectives[0] <= 500)


