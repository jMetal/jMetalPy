import logging
import warnings
from typing import TypeVar, List, Tuple

from bokeh.embed import file_html
from bokeh.events import DoubleTap
from bokeh.resources import CDN
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from bokeh.client import ClientSession
from bokeh.io import curdoc, reset_output
from bokeh.layouts import column
from bokeh.models import HoverTool, ColumnDataSource, TapTool, CustomJS, WheelZoomTool, Title
from bokeh.plotting import Figure

from jmetal.core.solution import Solution

warnings.filterwarnings("ignore", ".*GUI is implemented.*")

logger = logging.getLogger(__name__)

S = TypeVar('S')
SUPPORTED_FORMATS = ["eps", "jpeg", "jpg", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz", "tif", "tiff"]

"""
.. module:: graphics
   :platform: Unix, Windows
   :synopsis: Class for plotting solutions.

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Plot:

    def __init__(self, plot_title: str, reference_solution_list: List[S], number_of_objectives: int):
        """ Creates a new :class:`ScatterPlot` instance. Suitable for problems with 2 or 3 objectives.

        :param plot_title: Title of the scatter diagram.
        :param reference_solution_list: Reference solution list (if any).
        :param number_of_objectives: Number of objectives to be used (2D/3D).
        """
        self.plot_title = plot_title
        self.number_of_objectives = number_of_objectives
        self.reference_solution_list = reference_solution_list

    def __initialize(self) -> None:
        pass

    def plot(self, solution_list: List[S], show: bool = False) -> None:
        pass

    def update(self, solution_list: List[S], subtitle: str = "", persistence: bool = True) -> None:
        pass

    def get_objectives(self, solution_list: List[S]) -> Tuple[list, list, list]:
        """ Get coords (x,y,z) from a solution_list.

        :return: A tuple with (x,y,z) values. The third might be empty if working with a problem with 2 objectives."""
        if solution_list is None:
            raise Exception("Solution list is none!")

        points = list(solution.objectives for solution in solution_list)

        x_values, y_values = [point[0] for point in points], [point[1] for point in points]
        z_values = []

        try:
            z_values = [point[2] for point in points]
        except IndexError:
            pass

        return x_values, y_values, z_values


class Scatter(Plot):

    def __init__(self, plot_title: str, reference_solution_list: List[S] = None, number_of_objectives: int = 2):
        """ Creates a new :class:`ScatterPlot` instance. Suitable for problems with 2 or 3 objectives.

        :param plot_title: Title of the scatter diagram.
        :param number_of_objectives: Number of objectives to be used (2D/3D).
        :param reference_solution_list: Reference solution list (if any).
        """
        super().__init__(plot_title, reference_solution_list, number_of_objectives)
        logger.warning("Scatter() will be deprecated in the future. Use ScatterBokeh() instead")

    def __initialize(self) -> None:
        """ Initialize the scatter plot for the first time. """
        logger.info("Generating plot...")

        # Initialize a plot
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('jMetalPy')
        self.sc = None

        if self.number_of_objectives == 2:
            self.axis = self.fig.add_subplot(111)

            # Stylize axis
            self.axis.spines['top'].set_visible(False)
            self.axis.spines['right'].set_visible(False)
            self.axis.get_xaxis().tick_bottom()
            self.axis.get_yaxis().tick_left()
        else:
            self.axis = Axes3D(self.fig)
            self.axis.autoscale(enable=True, axis='both')

        self.axis.set_autoscale_on(True)
        self.axis.autoscale_view(True, True, True)

        # Style options
        self.axis.grid(color='#f0f0f5', linestyle='-', linewidth=1, alpha=0.5)
        self.fig.suptitle(self.plot_title, fontsize=13)

    def __plot(self, x_values, y_values, z_values, color: str = '#98FB98', marker: str = 'o', msize: int = 3):
        if self.number_of_objectives == 2:
            self.sc, = self.axis.plot(x_values, y_values,
                                      color=color, marker=marker, markersize=msize, ls='None', picker=10)
        else:
            self.sc, = self.axis.plot(x_values, y_values, z_values,
                                      color=color, marker=marker, markersize=msize, ls='None', picker=10)

    def plot(self, solution_list: List[S], show: bool = False) -> None:
        """ Plot a solution list. If any reference solution list, plot it first with other color. """
        if self.reference_solution_list:
            ref_x_values, ref_y_values, ref_z_values = self.get_objectives(self.reference_solution_list)
            self.__plot(ref_x_values, ref_y_values, ref_z_values, color='#323232', marker='*')

        x_values, y_values, z_values = self.get_objectives(solution_list)
        self.__plot(x_values, y_values, z_values)

        if show:
            self.fig.canvas.mpl_connect('pick_event', lambda event: self.__pick_handler(event, solution_list))
            plt.show()

    def update(self, solution_list: List[S], subtitle: str = "", persistence: bool = True) -> None:
        """ Update a plot with new values.

        .. note:: The plot must be initialized first.

        :param solution_list: List of points.
        :param subtitle: Plot subtitle.
        :param persistence: Keep old points instead of replacing them on real-time plots. """
        if self.sc is None:
            raise Exception("Error while updating: Initialize plot first!")

        x_values, y_values, z_values = self.get_objectives(solution_list)

        if persistence:
            # Replace with new points
            self.sc.set_data(x_values, y_values)
        else:
            # Add new points
            self.__plot(x_values, y_values, z_values)

        # Also, add event handler
        event_handler = \
            self.fig.canvas.mpl_connect('pick_event', lambda event: self.__pick_handler(event, solution_list))

        if self.number_of_objectives == 3:
            self.sc.set_3d_properties(z_values)

        # Update title with new times and evaluations
        self.fig.suptitle(subtitle, fontsize=13)

        # Re-align the axis
        self.axis.relim()
        self.axis.autoscale_view(True, True, True)

        # Draw
        self.fig.canvas.draw()
        plt.pause(0.01)

        # Disconnect the pick event for the next update
        self.fig.canvas.mpl_disconnect(event_handler)

    def save(self, file_name: str, fmt: str = 'eps', dpi: int = 200):
        """ Save the plot in a file.

        :param file_name: File name (without format).
        :param fmt: Output format.
        :param dpi: Pixels density. """
        logger.info("Saving to file...")

        if fmt not in SUPPORTED_FORMATS:
            raise Exception("{0} is not a valid format! Use one of these instead: {0}".format(fmt, SUPPORTED_FORMATS))
        self.fig.savefig(file_name + '.' + fmt, format=fmt, dpi=dpi)

    def __retrieve_info(self, x_val: float, y_val: float, solution: Solution) -> None:
        """ Retrieve information about a solution object. """
        logger.info("Output file: " + '{0}-{1}'.format(x_val, y_val))
        with open('{0}-{1}'.format(x_val, y_val), 'w') as of:
            of.write(solution.__str__())

    def __pick_handler(self, event, solution_list: List[S]):
        """ Handler for picking points from the plot. """
        line, ind = event.artist, event.ind[0]
        x, y = line.get_xdata(), line.get_ydata()

        logger.info('Selected data point ({0}): ({1}, {2})'.format(ind, x[ind], y[ind]))

        sol = next((solution for solution in solution_list
                    if solution.objectives[0] == x[ind] and solution.objectives[1] == y[ind]), None)

        if sol is not None:
            self.__retrieve_info(x[ind], y[ind], sol)
        else:
            logger.warning("Solution is none.")
            return True


class ScatterBokeh(Plot):

    def __init__(self, plot_title: str, reference: List[S] = None, number_of_objectives: int = 2,
                 source: ColumnDataSource = None, port: int=5006, ws_url: str="localhost:5006"):
        super().__init__(plot_title, reference, number_of_objectives)

        if source is None:
            self.source = ColumnDataSource(data=dict(x=[], y=[], variables=[]))

        if number_of_objectives > 2:
            raise Exception("ScatterBokeh() can only serve 2D charts. Use Scatter() instead.")

        self.figure = None
        self.client = ClientSession(websocket_url="ws://{0}/ws".format(ws_url))
        self.port = port
        self.doc = curdoc()

        self.__initialize()

    def __initialize(self) -> None:
        """ Set-up tools for plot. """
        code = '''
            selected = source.selected['1d']['indices'][0]
            console.log(source.selected['1d']['indices'])
            console.log(source.data)
            console.log(source.data.variables[selected])
        '''

        callback = CustomJS(args=dict(source=self.source), code=code)
        self.plot_tools = [TapTool(callback=callback), WheelZoomTool(), 'save', 'pan',
                           HoverTool(tooltips=[("index", "$index"), ("(x,y)", "($x, $y)")])]

    def plot(self, solution_list: List[S], show: bool=True) -> None:
        logger.debug("Opening Bokeh application on http://localhost:{0}/".format(self.port))

        # This is important to purge data (if any) between calls
        reset_output()

        # Set-up figure
        self.figure = Figure(output_backend="webgl", sizing_mode='scale_width',
                             title=self.plot_title, tools=self.plot_tools)

        # Plot reference solution list (if any)
        if self.reference_solution_list:
            ref_x_values, ref_y_values, ref_z_values = self.get_objectives(self.reference_solution_list)
            self.figure.line(x=ref_x_values, y=ref_y_values, legend="reference", color="green")

        # Plot solution list
        self.figure.scatter(x='x', y='y', legend="solutions", fill_alpha=0.7, source=self.source)

        x_values, y_values, z_values = self.get_objectives(solution_list)
        self.source.stream({'x': x_values, 'y': y_values, 'variables': [solution.variables for solution in solution_list]})

        # Add to curdoc
        self.doc.add_root(column(self.figure))
        self.client.push(self.doc)

        if show:
            self.client.show()

    def update(self, solution_list: List[S], new_title: str="", persistence: bool=True) -> None:
        # Check if plot has not been initialized first
        if self.figure is None:
            self.plot(solution_list)

        self.figure.title.text = new_title
        x_values, y_values, z_values = self.get_objectives(solution_list)

        if persistence:
            self.source.stream({'x': x_values, 'y': y_values,
                                'variables': [solution.variables for solution in solution_list]},
                               rollover=len(solution_list))
        else:
            self.source.stream({'x': x_values, 'y': y_values,
                                'variables': [solution.variables for solution in solution_list]})

    def save(self, file_name: str):
        html = file_html(models=self.figure, resources=CDN)
        # todo Save to file

    def disconnect(self):
        if self.is_connected():
            self.client.close()

    def is_connected(self) -> bool:
        return self.client.connected
