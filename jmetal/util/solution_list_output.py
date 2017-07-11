import logging
from typing import TypeVar, List, Generic, Tuple

from jmetal.util.graphic import ScatterPlot
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

S = TypeVar('S')


def get_data_points(solution_list: List[S]) -> Tuple[list, list]:
    points = list(sol.objectives for sol in solution_list)
    x_values, y_values = [x[0] for x in points], [y[1] for y in points]

    return x_values, y_values


class SolutionListOutput(Generic[S]):
    @staticmethod
    def plot_scatter(file_name, solution_list:List[S]):
        """ Plot non-dominated solutions. For problems with TWO variables.
        """
        x_values, y_values = get_data_points(solution_list)

        sc = ScatterPlot(plot_title="Pareto frontier")
        sc.simple_plot(x_values, y_values, save=True, file_name=file_name)

    @staticmethod
    def plot_scatter_real_time(solution_list: List[S], animation_speed: float):
        """ Plot non-dominated solutions in real-time. For problems with TWO variables.
        """
        global sc
        x_values, y_values = get_data_points(solution_list)

        if not plt.get_fignums():
            # The first time, set up plot
            sc = ScatterPlot(plot_title="Pareto frontier", animation_speed=animation_speed)
        else:
            sc.live_plot(x_values, y_values, solution_list)

    @staticmethod
    def print_variables_to_screen(solution_list:List[S]):
        for solution in solution_list:
            print(solution.variables[0])

    @staticmethod
    def print_function_values_to_screen(solution_list:List[S]):
        for solution in solution_list:
            print(str(solution_list.index(solution)) + ": ", sep='  ', end='', flush=True)
            print(solution.objectives, sep='  ', end='', flush=True)
            print()

    @staticmethod
    def print_function_values_to_file(file_name, solution_list:List[S]):
        logger.info("Output file (function values): " + file_name)
        with open(file_name, 'w') as of:
            for solution in solution_list:
                for function_value in solution.objectives:
                    of.write(str(function_value) + " ")
                of.write("\n")
