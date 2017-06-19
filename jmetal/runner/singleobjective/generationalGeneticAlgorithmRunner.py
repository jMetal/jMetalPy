from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.core.solution import BinarySolution, FloatSolution
from jmetal.operator.crossover import SinglePoint, SBX
from jmetal.operator.mutation import BitFlip, Polynomial
from jmetal.operator.selection import BinaryTournament
from jmetal.problem.singleobjectiveproblem import OneMax, Sphere

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    binary_example()
    float_example()
    run_as_a_thread_example()
    float_example_changing_the_stopping_condition()


def binary_example() -> None:
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


def run_as_a_thread_example() -> None:
    variables = 10
    problem = Sphere(variables)
    algorithm = GenerationalGeneticAlgorithm[FloatSolution, FloatSolution](
        problem,
        population_size = 100,
        max_evaluations = 25000,
        mutation = Polynomial(1.0/variables, distribution_index=20),
        crossover = SBX(1.0, distribution_index=20),
        selection = BinaryTournament())

    algorithm.start()
    logger.info("Algorithm (running as a thread): " + algorithm.get_name())
    logger.info("Problem: " + problem.get_name())

    algorithm.join()
    result = algorithm.get_result()
    logger.info("Solution: " + str(result.variables))
    logger.info("Fitness:  " + str(result.objectives[0]))
    logger.info("Computing time: " + str(algorithm.total_computing_time))


def float_example() -> None:
    variables = 10
    problem = Sphere(variables)
    algorithm = GenerationalGeneticAlgorithm[FloatSolution, FloatSolution](
        problem,
        population_size = 100,
        max_evaluations = 25000,
        mutation = Polynomial(1.0/variables, distribution_index=20),
        crossover = SBX(1.0, distribution_index=20),
        selection = BinaryTournament())

    algorithm.run()
    result = algorithm.get_result()
    logger.info("Algorithm (continuous problem): " + algorithm.get_name())
    logger.info("Problem: " + problem.get_name())
    logger.info("Solution: " + str(result.variables))
    logger.info("Fitness:  " + str(result.objectives[0]))
    logger.info("Computing time: " + str(algorithm.total_computing_time))


def float_example_changing_the_stopping_condition() -> None:
    variables = 10
    problem = Sphere(variables)

    class GGA2(GenerationalGeneticAlgorithm[FloatSolution, FloatSolution]):
        def is_stopping_condition_reached(self):
            return self.get_current_computing_time() > 2

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
