import logging
from typing import TypeVar, List, Generic

import matplotlib.pyplot as plt

from jmetal.util.graphic import ScatterPlot

logger = logging.getLogger(__name__)

S = TypeVar('S')


class SolutionListOutput(Generic[S]):

    @staticmethod
    def plot_frontier_to_file(solution_list: List[S], reference_solution_list: List[S], title: str,
                              file_name: str, output_format: str='eps'):
        """ Save plot of non-dominated solutions to file. For problems with TWO or THREE variables """
        sc = ScatterPlot(title, solution_list[0].number_of_objectives, reference_solution_list)
        sc.plot(solution_list, False)
        sc.save(file_name, output_format)

    @staticmethod
    def plot_frontier_to_screen(solution_list: List[S], reference_solution_list: List[S], title: str):
        """ Plot non-dominated solutions interactively. For problems with TWO or THREE variables """
        sc = ScatterPlot(title, solution_list[0].number_of_objectives, reference_solution_list)
        sc.plot(solution_list, True)

    @staticmethod
    def plot_frontier_live(solution_list: List[S], reference_solution_list: List[S], title: str,
                           evaluations: int, computing_time: float, replace: bool):
        """ Plot non-dominated solutions in real-time. For problems with TWO or THREE variables """
        global sc

        if not plt.get_fignums():
            # The first time, set up plot
            sc = ScatterPlot(title, solution_list[0].number_of_objectives, reference_solution_list)
            sc.plot(solution_list)
        else:
            subtitle = '{0}, Eval: {1}, Time: {2}'.format(title, evaluations, computing_time)
            sc.update(solution_list, subtitle, replace)

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
    def print_function_values_to_file(solution_list: List[S], file_name):
        logger.info("Output file (function values): " + file_name)
        with open(file_name, 'w') as of:
            for solution in solution_list:
                for function_value in solution.objectives:
                    of.write(str(function_value) + " ")
                of.write("\n")
