import logging
from typing import TypeVar, List, Generic

import matplotlib.pyplot as plt

from jmetal.util.graphic import ScatterPlot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

S = TypeVar('S')


class SolutionListOutput(Generic[S]):

    @staticmethod
    def plot_frontier_to_file(solution_list: List[S], reference_solution_list: List[S],
                              file_name="output", output_format: str='eps', dpi: int=200):
        """ Plot non-dominated solutions. For problems with TWO variables """
        sc = ScatterPlot("Pareto frontier")
        sc.plot(solution_list, reference_solution_list, True, output_format, dpi, file_name)

    @staticmethod
    def plot_frontier_to_screen(solution_list: List[S]):
        """ Plot non-dominated solutions. For problems with TWO variables """
        sc = ScatterPlot("Pareto frontier")
        sc.plot_interactive(solution_list)

    @staticmethod
    def plot_frontier_interactive(solution_list: List[S], reference_solution_list: List[S], evaluations: int,
                                  computing_time: float, animation_speed: float=1 * 10e-10):
        """ Plot non-dominated solutions in real-time. For problems with TWO variables """
        global sc

        if not plt.get_fignums():
            # The first time, set up plot
            sc = ScatterPlot("Pareto frontier", animation_speed)
            sc.plot(solution_list, reference_solution_list, save=False)
        else:
            sc.update(solution_list, evaluations, computing_time)

    @staticmethod
    def print_variables_to_screen(solution_list: List[S]):
        for solution in solution_list:
            print(solution.variables[0])

    @staticmethod
    def print_function_values_to_screen(solution_list: List[S]):
        for solution in solution_list:
            print(str(solution_list.index(solution)) + ": ", sep='  ', end='', flush=True)
            print(solution.objectives, sep='  ', end='', flush=True)
            print()

    @staticmethod
    def print_function_values_to_file(file_name, solution_list: List[S]):
        logger.info("Output file (function values): " + file_name)
        with open(file_name, 'w') as of:
            for solution in solution_list:
                for function_value in solution.objectives:
                    of.write(str(function_value) + " ")
                of.write("\n")
