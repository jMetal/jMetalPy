import logging
import os
from pathlib import Path
from typing import List

from jmetal.core.solution import FloatSolution, Solution
from jmetal.util.archive import Archive, NonDominatedSolutionsArchive

logger = logging.getLogger(__name__)


"""
.. module:: solutions
   :platform: Unix, Windows
   :synopsis: Utils to print solutions.

.. moduleauthor:: Antonio J. Nebro <ajnebro@uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


def get_non_dominated_solutions(solutions: List[Solution]) -> List[Solution]:
    archive: Archive = NonDominatedSolutionsArchive()

    for solution in solutions:
        archive.add(solution)

    return archive.solution_list


def read_solutions(filename: str) -> List[FloatSolution]:
    """Reads a reference front from a file.

    :param filename: File path where the front is located.
    """
    front = []

    if Path(filename).is_file():
        with open(filename) as file:
            for line in file:
                vector = [float(x) for x in line.split()]

                solution = FloatSolution([], [], len(vector))
                solution.objectives = vector

                front.append(solution)
    else:
        logger.warning("Reference front file was not found at {}".format(filename))

    return front


def print_variables_to_file(solutions, filename: str):
    logger.info("Output file (variables): " + filename)

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass

    if type(solutions) is not list:
        solutions = [solutions]

    with open(filename, "w") as of:
        for solution in solutions:
            for variables in solution.variables:
                of.write(str(variables) + " ")
            of.write("\n")


def print_variables_to_screen(solutions):
    if type(solutions) is not list:
        solutions = [solutions]

    for solution in solutions:
        print(solution.variables[0])


def print_function_values_to_file(solutions, filename: str):
    logger.info("Output file (function values): " + filename)

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass

    if type(solutions) is not list:
        solutions = [solutions]

    with open(filename, "w") as of:
        for solution in solutions:
            for function_value in solution.objectives:
                of.write(str(function_value) + " ")
            of.write("\n")


def print_function_values_to_screen(solutions):
    if type(solutions) is not list:
        solutions = [solutions]

    for solution in solutions:
        print(str(solutions.index(solution)) + ": ", sep="  ", end="", flush=True)
        print(solution.objectives, sep="  ", end="", flush=True)
        print()
