import logging
from pathlib import Path
from typing import List

from jmetal.core.solution import FloatSolution

LOGGER = logging.getLogger('jmetal')

"""
.. module:: solution_list
   :platform: Unix, Windows
   :synopsis: Utils to print solutions.

.. moduleauthor:: Antonio J. Nebro <ajnebro@uma.es>
"""


def read_front(file_path: str) -> List[FloatSolution]:
    """ Reads a reference front from a file.

    :param file_path: File path where the front is located.
    """
    front = []

    if Path(file_path).is_file():
        with open(file_path) as file:
            for line in file:
                vector = [float(x) for x in line.split()]

                solution = FloatSolution(2, 2, 0, [], [])
                solution.objectives = vector

                front.append(solution)
    else:
        LOGGER.warning('Reference front file was not found at {}'.format(file_path))

    return front


def print_variables_to_file(solution_list: list, file_name):
    LOGGER.info('Output file (variables): ' + file_name)
    with open(file_name, 'w') as of:
        for solution in solution_list:
            for variables in solution.variables:
                of.write(str(variables) + " ")
            of.write("\n")


def print_variables_to_screen(solution_list: list):
    for solution in solution_list:
        print(solution.variables[0])


def print_function_values_to_screen(solution_list: list):
    for solution in solution_list:
        print(str(solution_list.index(solution)) + ": ", sep='  ', end='', flush=True)
        print(solution.objectives, sep='  ', end='', flush=True)
        print()


def print_function_values_to_file(solution_list: list, file_name):
    LOGGER.info('Output file (function values): ' + file_name)
    with open(file_name, 'w') as of:
        for solution in solution_list:
            for function_value in solution.objectives:
                of.write(str(function_value) + " ")
            of.write("\n")
