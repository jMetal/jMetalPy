import logging
import warnings
from typing import TypeVar, List, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from jmetal.core.solution import Solution

warnings.filterwarnings("ignore",".*GUI is implemented.*")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

S = TypeVar('S')
SUPPORTED_FORMATS = ["eps", "jpeg", "jpg", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz", "tif", "tiff"]

"""
.. module:: graphics
   :platform: Unix, Windows
   :synopsis: Class for plotting solutions.

.. moduleauthor:: Antonio Benítez <antonio.b@uma.es>
"""


class ScatterPlot:

    def __init__(self, plot_title: str, number_of_objectives: int=2):
        """ Creates a new :class:`ScatterPlot` instance. Suitable for problems with 2 or 3 objectives.

        :param plot_title: Title of the scatter diagram.
        :param number_of_objectives: Number of objectives to be used (2D/3D).
        """
        self.plot_title = plot_title
        self.number_of_objectives = number_of_objectives

        # Initialize a plot
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('jMetalPy')
        self.sc = None

        if number_of_objectives == 2:
            self.axis = self.fig.add_subplot(111)
        else:
            self.axis = Axes3D(self.fig)
            self.axis.autoscale(enable=True, axis='both')

        self.__initialize()

    def __initialize(self) -> None:
        """ Initialize the scatter plot the first time. """
        logger.info("Generating plot...")

        self.axis.set_autoscale_on(True)
        self.axis.autoscale_view(True, True, True)

        # Style options
        self.axis.grid(color='#f0f0f5', linestyle='-', linewidth=2, alpha=0.5)
        self.fig.suptitle(self.plot_title, fontsize=13)

    def __plot(self, x_values, y_values, z_values, solution_list, color: str='go'):
        if self.number_of_objectives == 2:
            self.sc, = self.axis.plot(x_values, y_values, color, markersize=3, picker=10)
            self.fig.canvas.mpl_connect('pick_event', lambda event: self.__pick_handler(event, solution_list))
        else:
            self.sc, = self.axis.plot(x_values, y_values, z_values, color, markersize=3, picker=10)

    def plot(self, solution_list: List[S], reference_solution_list: List[S], show: bool=False) -> None:
        if reference_solution_list:
            ref_x_values, ref_y_values, ref_z_values = self.__get_objectives(reference_solution_list)
            self.__plot(ref_x_values, ref_y_values, ref_z_values, None, 'b')

        x_values, y_values, z_values = self.__get_objectives(solution_list)
        self.__plot(x_values, y_values, z_values, solution_list)

        if show:
            plt.show()

    def update(self, solution_list: List[S], evaluations: int=0, computing_time: float=0) -> None:
        """ Update a plot with new values.

        .. note:: The plot must be initialized first. """
        if self.sc is None:
            raise Exception("Error while updating: Initialize plot first!")

        x_values, y_values, z_values = self.__get_objectives(solution_list)

        # Replace with new points
        self.sc.set_data(x_values, y_values)

        # Also, add event handler
        event_handler = \
            self.fig.canvas.mpl_connect('pick_event', lambda event: self.__pick_handler(event, solution_list))

        if self.number_of_objectives == 3:
            self.sc.set_3d_properties(z_values)

        # Update title with new times and evaluations
        self.fig.suptitle('{0}, Eval: {1}, Time: {2}'.format(self.plot_title, evaluations, computing_time), fontsize=13)

        # Re-align the axis
        self.axis.relim()
        self.axis.autoscale_view(True, True, True)

        # Draw
        self.fig.canvas.draw()
        plt.pause(0.01)

        # Disconnect the pick event for the next update
        self.fig.canvas.mpl_disconnect(event_handler)

    def save(self, file_name: str, fmt: str='eps', dpi: int=200):
        """ Save the plot in a file. """
        logger.info("Saving to file...")

        if fmt not in SUPPORTED_FORMATS:
            raise Exception("{0} is not a valid format! Use one of these instead: {0}".format(fmt, SUPPORTED_FORMATS))
        self.fig.savefig(file_name + '.' + fmt, format=fmt, dpi=dpi)

    def retrieve_info(self, x_val: float, y_val: float, solution: Solution) -> None:
        """ Retrieve some information about a solution object. """
        logger.info("Output file: " + '{0}-{1}'.format(x_val, y_val))
        with open('{0}-{1}'.format(x_val, y_val), 'w') as of:
            for function_value in solution.objectives:
                of.write(str(function_value) + " ")
            of.write("\n \n")
            for function_variable in solution.variables:
                of.write(str(function_variable) + "\n")
            of.write("\n")

    def __get_objectives(self, solution_list: List[S]) -> Tuple[list, list, list]:
        """ Get coords (x,y,z) from a solution_list.

        :return: A tuple with (x,y,z) values. The third might be empty if working with a problem with 2 objectives."""
        if solution_list is None:
            raise Exception("Solution list is none!")

        points = list(solution.objectives for solution in solution_list)

        x_values, y_values = [point[0] for point in points], [point[1] for point in points]
        z_values = []

        if self.number_of_objectives == 3:
            z_values = [point[2] for point in points]

        return x_values, y_values, z_values

    def __get_solution_from_list(self, solution_list: List[S], x_val: float, y_val: float) -> None:
        """ :return: A solution object associated with some values of (x,y) """
        sol = next((solution for solution in solution_list
                    if solution.objectives[0] == x_val and solution.objectives[1] == y_val), None)

        if sol is not None:
            self.retrieve_info(x_val, y_val, sol)
        else:
            logger.warning("Solution is none.")

    def __pick_handler(self, event, solution_list: List[S]):
        """ Handler for picking points from the plot. """
        line, ind = event.artist, event.ind[0]
        x, y = line.get_xdata(), line.get_ydata()

        logger.info('Selected data point ({0}): ({1}, {2})'.format(ind, x[ind], y[ind]))
        self.__get_solution_from_list(solution_list, x[ind], y[ind])
