import logging

from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.core.solution import BinarySolution, FloatSolution
from jmetal.operator.crossover import SinglePoint, SBX
from jmetal.operator.mutation import BitFlip, Polynomial
from jmetal.operator.selection import BinaryTournament
from jmetal.problem.singleobjectiveproblem import OneMax, Sphere

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def gGABinary() -> None:
    bits = 512
    problem = OneMax(bits)
    algorithm = GenerationalGeneticAlgorithm[BinarySolution, BinarySolution](
        problem,
        population_size = 100,
        max_evaluations = 25000,
        mutation = BitFlip(1.0/bits),
        crossover = SinglePoint(0.9),
        selection = BinaryTournament())

    algorithm.run()
    result = algorithm.get_result()
    logger.info("Algorithm (binary problem): " + algorithm.get_name())
    logger.info("Problem: " + problem.get_name())
    logger.info("Solution: " + str(result.variables[0]))
    logger.info("Fitness:  " + str(result.objectives[0]))
    logger.info("Computing time: " + str(algorithm.total_computing_time))


if __name__ == '__main__':
    gGABinary()
