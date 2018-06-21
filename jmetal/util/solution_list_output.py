import logging
from typing import TypeVar, Generic

from jmetal.util.graphic import ScatterBokeh

S = TypeVar('S')
logger = logging.getLogger(__name__)


class GraphicSolutionList(Generic[S]):

    def __init__(self, title: str, reference: list=None, number_of_objectives: int=2):
        self.bokeh = ScatterBokeh(title, reference, number_of_objectives)

    def plot_frontier_to_file(self, solution_list: list, file_name: str):
        """ Save plot of non-dominated solutions to file. For problems with TWO or THREE variables """
        self.bokeh.plot(solution_list)
        self.bokeh.save(file_name)

    def plot_frontier_to_screen(self, solution_list: list):
        """ Plot non-dominated solutions interactively. For problems with TWO or THREE variables """
        self.bokeh.plot(solution_list, show=True)

    def plot_frontier_live(self, solution_list: list, new_title: str, replace: bool):
        """ Plot non-dominated solutions in real-time. For problems with TWO or THREE variables """
        self.bokeh.update(solution_list, new_title, replace)


class PrintSolutionList(Generic[S]):

    @staticmethod
    def print_variables_to_screen(solution_list: list):
        for solution in solution_list:
            print(solution.variables[0])

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
