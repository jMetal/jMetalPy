import logging

from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBX
from jmetal.operator.mutation import Polynomial
from jmetal.operator.selection import BinaryTournament
from jmetal.problem.singleobjective.unconstrained import Sphere

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    variables = 10
    problem = Sphere(variables)

    class GGA2(GenerationalGeneticAlgorithm[FloatSolution, FloatSolution]):
        def is_stopping_condition_reached(self):
            return self.get_current_computing_time() > 3

    algorithm = GGA2(
        problem,
        population_size = 100,
        max_evaluations=0,
        mutation = Polynomial(1.0/variables, distribution_index=20),
        crossover = SBX(1.0, distribution_index=20),
        selection = BinaryTournament())

    algorithm.run()
    result = algorithm.get_result()
    logger.info("Algorithm (stop for timeout): " + algorithm.get_name())
    logger.info("Problem: " + problem.get_name())
    logger.info("Solution: " + str(result.variables))
    logger.info("Fitness:  " + str(result.objectives[0]))
    logger.info("Computing time: " + str(algorithm.total_computing_time))


if __name__ == '__main__':
    main()
