import logging
import os
from pathlib import Path

import moocore
import numpy as np

from jmetal.core.solution import FloatSolution, Solution

logger = logging.getLogger(__name__)


"""
.. module:: solutions
   :platform: Unix, Windows
   :synopsis: Utils to print solutions.

.. moduleauthor:: Antonio J. Nebro <ajnebro@uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


def get_non_dominated_solutions(solutions: list[Solution]) -> list[Solution]:
    """Filter a list of solutions down to its non-dominated subset.

    Delegates to `moocore.is_nondominated` for efficiency: the previous
    implementation added solutions one at a time to a `NonDominatedSolutionsArchive`,
    which is O(n) per insertion (O(n^2) overall) -- noticeably slow for the
    thousands of solutions an unbounded external archive can accumulate.
    `moocore.is_nondominated` filters the whole batch at once and already treats
    solutions with identical objectives as duplicates, keeping only the first
    occurrence -- the same behavior `NonDominatedSolutionsArchive` had.

    Args:
        solutions: The solutions to filter.

    Returns:
        The non-dominated subset, in their original relative order.
    """
    if not solutions:
        return []

    objectives = np.array([solution.objectives for solution in solutions], dtype=float)
    keep = moocore.is_nondominated(objectives)

    return [solution for solution, is_kept in zip(solutions, keep, strict=True) if is_kept]


def read_solutions(filename: str) -> list[FloatSolution]:
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
        logger.warning(f"Reference front file was not found at {filename}")

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
