import logging

from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.core.solution import BinarySolution
from jmetal.operator.crossover import SinglePoint
from jmetal.operator.mutation import BitFlip
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.problem.singleobjective.unconstrained import OneMax

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    bits = 256
    problem = OneMax(bits)
    algorithm = GenerationalGeneticAlgorithm[BinarySolution, BinarySolution](
        problem,
        population_size = 100,
        max_evaluations = 150000,
        mutation = BitFlip(1.0/bits),
        crossover = SinglePoint(0.9),
        selection = BinaryTournamentSelection())

    algorithm.run()
    result = algorithm.get_result()

    logger.info("Algorithm (binary problem): " + algorithm.get_name())
    logger.info("Problem: " + problem.get_name())
    logger.info("Solution: " + str(result.variables[0]))
    logger.info("Fitness:  " + str(result.objectives[0]))

if __name__ == '__main__':
    main()
