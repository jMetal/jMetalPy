from typing import List, TypeVar

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.core.solution import BinarySolution, FloatSolution
from jmetal.operator.crossover import SinglePoint, SBX
from jmetal.operator.mutation import BitFlip, Polynomial
from jmetal.operator.selection import BinaryTournament
from jmetal.problem.multiobjectiveproblem import Fonseca, Kursawe, Schaffer

import logging

from jmetal.util.solution_list_output import SolutionListOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    nsgaii_runner()


def nsgaii_runner() -> None:
    problem = Kursawe()
    algorithm = NSGAII[FloatSolution, List[FloatSolution]](
        problem,
        population_size = 100,
        max_evaluations = 25000,
        mutation = Polynomial(1.0/problem.number_of_variables, distribution_index=20),
        crossover = SBX(1.0, distribution_index=20),
        selection = BinaryTournament())

    algorithm.run()
    result = algorithm.get_result()

    SolutionListOutput[FloatSolution].print_function_values_to_screen(result)
    SolutionListOutput[FloatSolution].print_function_values_to_file("FUN."+problem.get_name(), result)

    logger.info("Algorithm (continuous problem): " + algorithm.get_name())
    logger.info("Problem: " + problem.get_name())
    logger.info("Computing time: " + str(algorithm.total_computing_time))

if __name__ == '__main__':
    main()
