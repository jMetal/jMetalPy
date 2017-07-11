import logging
from typing import TypeVar, List

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

S = TypeVar('S')

"""
.. module:: graphics
   :platform: Unix, Windows
   :synopsis: Class for plotting solutions.

.. moduleauthor:: Antonio Benítez <antonio.b@uma.es>
"""


class ScatterPlot():
    """ Scatter plot.
    """
    def __init__(self, plot_title: str, animation_speed: float = 1*10e-10):
        """ Creates a new :class:`ScatterPlot` instance.
        Args:
           plot_title (str): Title of the scatter diagram.
           animation_speed (float): Delay (for live plot only). Allow time for the gui event loops to trigger
           and update the display.
        """
        self.plot_title = plot_title
        self.fig = plt.figure()
        self.axis = self.fig.add_subplot(111)
        self.sc = None

        # Real-time plotting options
        self.animation_speed = animation_speed

    def _init_plot(self, is_auto_scalable: bool) -> None:
        if is_auto_scalable:
            self.axis.set_autoscale_on(True)
            self.axis.autoscale_view(True, True, True)

        logger.info("Generating plot...")

        # Style options
        self.axis.grid(color='#f0f0f5', linestyle='-', linewidth=2, alpha=0.5)
        self.fig.suptitle(self.plot_title, fontsize=14, fontweight='bold')

    def simple_plot(self, x_values: list, y_values: list, file_name: str = "output",
                    format: str = 'png', dpi: int = 200, save: bool = False) -> None:
        self._init_plot(is_auto_scalable=True)
        self.sc, = self.axis.plot(x_values, y_values, 'bo', markersize=7, picker=7)

        if save:
            # Supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            logger.info("Output file (function plot): " + file_name + '.' + format)
            self.fig.savefig(file_name + '.' + format, format=format, dpi=dpi)

    def __search_solution(self, solution_list: List[S], x_val: float, y_val: float) -> None:
        """ Return a solution object associated with some values of (x,y). """
        for solution in solution_list:
            if solution.objectives[0] == x_val and solution.objectives[1] == y_val:
                logger.info('Solution associated to ({0}, {1}): {2}'
                            .format(x_val, y_val, solution))

    def live_plot(self, x_values: list, y_values: list, solution_list: List[S]) -> None:
        def pick_handler(event):
            x, y = event.mouseevent.xdata, event.mouseevent.ydata

            logger.info('Selected data point: ({0}, {1})'.format(x, y))
            self.__search_solution(solution_list, x, y)

        if not self.sc:
            # The first time, initialize plot and add mouse event
            self.fig.canvas.mpl_connect('pick_event', pick_handler)
            self.simple_plot(x_values, y_values)

        # Update
        self.sc.set_data(x_values, y_values)

        self.axis.relim()
        self.axis.autoscale_view(True, True, True)

        plt.draw()
        plt.pause(self.animation_speed)