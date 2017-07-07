import logging
from typing import TypeVar, List, Generic

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

S = TypeVar('S')


class SolutionListOutput(Generic[S]):
    @staticmethod
    def plot_scatter(file_name, solution_list:List[S]):
        """ Plot non-dominated solutions.
        """
        sol = []
        for solution in solution_list:
            for function_value in solution.objectives:
                sol.append(function_value)

        values = [tuple(sol[i:i+2]) for i in range(0, len(sol), 2)]

        x_val = [x[0] for x in values]
        y_val = [x[1] for x in values]

        logger.info("Output file (function plot): " + file_name + '.png')
        plt.title('Pareto frontier')
        plt.scatter(x_val, y_val, s=10)
        plt.savefig(file_name + '.png', dpi=200)

    @staticmethod
    def plot_scatter_real_time(solution_list:List[S], animation_speed: float = 1*10e-10):
        """ Plot non-dominated solutions in real-time.
        """

        def get_data():
            sol = []
            for solution in solution_list:
                for function_value in solution.objectives:
                    sol.append(function_value)

            values = [tuple(sol[i:i + 2]) for i in range(0, len(sol), 2)]

            x_val = [x[0] for x in values]
            y_val = [x[1] for x in values]

            yield x_val, y_val

        def init_plot():
            logger.info("Generating plot...")
            x_val, y_val = next(get_data())

            # Setup plot
            plt.ion()
            fig = plt.figure()
            axes = fig.add_subplot(111)
            axes.set_autoscale_on(True)
            axes.autoscale_view(True, True, True)
            sc, = axes.plot(x_val, y_val, 'ro')

            plt.show()

            return sc, axes

        global figure, axes

        if not plt.get_fignums():
            # The first time, set up plot
            figure, axes = init_plot()
        else:
            x_val, y_val = next(get_data())

            figure.set_data(x_val, y_val)

            axes.relim()
            axes.autoscale_view(True, True, True)

            plt.draw()
            plt.pause(animation_speed)

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
