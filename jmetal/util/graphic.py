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
        """ Creates a new :class:`Plot` instance.

        :param plot_title: Title of the scatter diagram.
        """
        self.plot_title = plot_title
        self.number_of_objectives = number_of_objectives

        # Initialize a plot
        self.fig = plt.figure()
        self.sc = None

        if number_of_objectives == 2:
            self.axis = self.fig.add_subplot(111)
        else:
            self.axis = Axes3D(self.fig)
            self.axis.autoscale(enable=True, axis='both')

        self.__initialize()

    def __initialize(self, is_auto_scalable: bool = True) -> None:
        """ Initialize the scatter plot the first time. """
        if is_auto_scalable:
            self.axis.set_autoscale_on(True)
            self.axis.autoscale_view(True, True, True)

        logger.info("Generating plot...")

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
        """ Update a plot(). Note that the plot must be initialized first. """
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
        if fmt not in SUPPORTED_FORMATS:
            raise Exception("{0} is not a valid format! Use one of these instead: {0}".format(fmt, SUPPORTED_FORMATS))
        self.fig.savefig(file_name + '.' + fmt, format=fmt, dpi=dpi)

    def retrieve_info(self, solution: Solution) -> None:
        """ Retrieve more information about a solution object. """
        pass

    def __get_objectives(self, solution_list: List[S]) -> Tuple[list, list, list]:
        """ Get coords (x,y) from a solution_list. """
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
            self.retrieve_info(sol)
        else:
            logger.warning("Solution is none.")

    def __pick_handler(self, event, solution_list: List[S]):
        """ Handler for picking points from the plot. """
        line, ind = event.artist, event.ind[0]
        x, y = line.get_xdata(), line.get_ydata()

        logger.info('Selected data point ({0}): ({1}, {2})'.format(ind, x[ind], y[ind]))
        self.__get_solution_from_list(solution_list, x[ind], y[ind])
