import logging
from typing import TypeVar, List, Generic

import matplotlib.pyplot as plt

from jmetal.util.graphic import ScatterPlot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

S = TypeVar('S')


class SolutionListOutput(Generic[S]):
    @staticmethod
    def plot_scatter_to_file(solution_list: List[S], file_name, output_format: str, dpi: int,
                             plot_title="Pareto frontier"):
        """ Plot non-dominated solutions. For problems with TWO variables.
        """
        sc = ScatterPlot(plot_title=plot_title)
        sc.simple_plot(solution_list=solution_list, file_name=file_name, fmt=output_format, dpi=dpi)

    @staticmethod
    def plot_scatter_to_screen(solution_list: List[S],
                               plot_title="Pareto frontier (interactive)"):
        """ Plot non-dominated solutions. For problems with TWO variables.
        """
        sc = ScatterPlot(plot_title=plot_title)
        sc.interactive_plot(solution_list=solution_list)

    @staticmethod
    def plot_scatter_real_time(solution_list: List[S], evaluations: int, computing_time: float, animation_speed: float,
                               plot_title="Pareto frontier (real-time)"):
        """ Plot non-dominated solutions in real-time. For problems with TWO variables.
        """
        global sc

        if not plt.get_fignums():
            # The first time, set up plot
            sc = ScatterPlot(plot_title=plot_title, animation_speed=animation_speed)
            sc.simple_plot(solution_list=solution_list, save=False)
        else:
            sc.update(solution_list=solution_list, evaluations=evaluations, computing_time=computing_time)

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
