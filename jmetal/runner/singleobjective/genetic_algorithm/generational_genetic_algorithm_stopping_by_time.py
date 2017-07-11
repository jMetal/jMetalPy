import logging

from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.core.solution import BinarySolution, FloatSolution
from jmetal.operator.crossover import SinglePoint, SBX
from jmetal.operator.mutation import BitFlip, Polynomial
from jmetal.operator.selection import BinaryTournament
from jmetal.problem.singleobjectiveproblem import OneMax, Sphere

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    class GGA2(GenerationalGeneticAlgorithm[FloatSolution, FloatSolution]):
        def is_stopping_condition_reached(self):
            # Re-define the stopping condition
            reached = [False, True][self.get_current_computing_time() > 4]

            if reached:
                logger.info("Stopping condition reached!")

            return reached

    variables = 10
    problem = Sphere(variables)
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


if __name__ == '__main__':
    main()
