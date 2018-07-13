import logging
from typing import TypeVar, Generic

logger = logging.getLogger(__name__)

S = TypeVar('S')

"""
.. module:: solution_list
   :platform: Unix, Windows
   :synopsis: Utils to print solutions.

.. moduleauthor:: Antonio J. Nebro <ajnebro@uma.es>
"""


class SolutionList(Generic[S]):

    @staticmethod
    def print_variables_to_screen(solution_list: list):
        for solution in solution_list:
            print(solution.variables[0])

    @staticmethod
    def print_variables_to_file(solution_list: list, file_name):
        logger.info("Output file (variables): " + file_name)
        with open(file_name, 'w') as of:
            for solution in solution_list:
                for variables in solution.variables:
                    of.write(str(variables) + " ")
                of.write("\n")

    @staticmethod
    def print_function_values_to_screen(solution_list: list):
        for solution in solution_list:
            print(str(solution_list.index(solution)) + ": ", sep='  ', end='', flush=True)
            print(solution.objectives, sep='  ', end='', flush=True)
            print()

    @staticmethod
    def print_function_values_to_file(solution_list: list, file_name):
        logger.info("Output file (function values): " + file_name)
        with open(file_name, 'w') as of:
            for solution in solution_list:
                for function_value in solution.objectives:
                    of.write(str(function_value) + " ")
                of.write("\n")
