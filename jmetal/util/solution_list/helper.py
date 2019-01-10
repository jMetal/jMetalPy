import logging
import os
import numpy
from pathlib import Path
from typing import List

from jmetal.core.solution import FloatSolution, Solution

LOGGER = logging.getLogger('jmetal')

"""
.. module:: solution_list
   :platform: Unix, Windows
   :synopsis: Utils to print solutions.

.. moduleauthor:: Antonio J. Nebro <ajnebro@uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


def read_solutions(file_path: str) -> List[FloatSolution]:
    """ Reads a reference front from a file.

    :param file_path: File path where the front is located.
    """
    front = []

    if Path(file_path).is_file():
        with open(file_path) as file:
            for line in file:
                vector = [float(x) for x in line.split()]

                solution = FloatSolution(2, 2, [], [])
                solution.objectives = vector

                front.append(solution)
    else:
        LOGGER.warning('Reference front file was not found at {}'.format(file_path))

    return front


def print_variables_to_file(solution_list: list, file_name):
    LOGGER.info('Output file (variables): ' + file_name)

    try:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    except FileNotFoundError:
        pass

    with open(file_name, 'w') as of:
        for solution in solution_list:
            for variables in solution.variables:
                of.write(str(variables) + " ")
            of.write("\n")


def print_variables_to_screen(solution_list: list):
    for solution in solution_list:
        print(solution.variables[0])


def print_function_values_to_file(solution_list: list, file_name):
    LOGGER.info('Output file (function values): ' + file_name)

    try:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    except FileNotFoundError:
        pass

    with open(file_name, 'w') as of:
        for solution in solution_list:
            for function_value in solution.objectives:
                of.write(str(function_value) + ' ')
            of.write('\n')


def print_function_values_to_screen(solution_list: list):
    for solution in solution_list:
        print(str(solution_list.index(solution)) + ": ", sep='  ', end='', flush=True)
        print(solution.objectives, sep='  ', end='', flush=True)
        print()


def get_numpy_array_from_objectives(solution_list: List[Solution]):
    list_of_objectives = []
    for solution in solution_list:
        list_of_objectives.append(solution.objectives)

    return numpy.array(list_of_objectives)


