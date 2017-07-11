import logging
from typing import TypeVar, List, Generic, Tuple

from jmetal.util.graphic import ScatterPlot
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

S = TypeVar('S')


class SolutionListOutput(Generic[S]):
    @staticmethod
    def plot_scatter_to_file(file_name, solution_list:List[S]):
        """ Plot non-dominated solutions. For problems with TWO variables.
        """
        sc = ScatterPlot(plot_title="Pareto frontier")
        sc.simple_plot(solution_list, file_name=file_name)

    @staticmethod
    def plot_scatter_to_screen(solution_list:List[S]):
        """ Plot non-dominated solutions. For problems with TWO variables.
        """
        sc = ScatterPlot(plot_title="Pareto frontier (interactive)")
        sc.interactive_plot(solution_list)

    @staticmethod
    def plot_scatter_real_time(solution_list: List[S], animation_speed: float):
        """ Plot non-dominated solutions in real-time. For problems with TWO variables.
        """
        global sc

        if not plt.get_fignums():
            # The first time, set up plot
            sc = ScatterPlot(plot_title="Pareto frontier (real-time)", animation_speed=animation_speed)
            sc.simple_plot(solution_list, save=False)
        else:
            sc.update(solution_list)

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
