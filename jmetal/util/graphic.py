import logging
import warnings
from typing import TypeVar, List, Tuple

from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.client import ClientSession
from bokeh.io import curdoc, reset_output
from bokeh.layouts import column, row
from bokeh.models import HoverTool, ColumnDataSource, TapTool, CustomJS, WheelZoomTool
from bokeh.plotting import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from jmetal.core.solution import Solution

warnings.filterwarnings("ignore", ".*GUI is implemented.*")

logger = logging.getLogger(__name__)
S = TypeVar('S')

"""
.. module:: Visualization
   :platform: Unix, Windows
   :synopsis: Classes for plotting solutions.

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Plot:

    def __init__(self, plot_title: str, number_of_objectives: int,
                 xaxis_label: str='', yaxis_label: str='', zaxis_label: str=''):
        self.plot_title = plot_title
        self.number_of_objectives = number_of_objectives

        self.xaxis_label = xaxis_label
        self.yaxis_label = yaxis_label
        self.zaxis_label = zaxis_label

    def get_objectives(self, solution_list: List[S]) -> Tuple[list, list, list]:
        if solution_list is None:
            raise Exception("Solution list is none!")

        points = list(solution.objectives for solution in solution_list)

        x_values, y_values = [point[0] for point in points], [point[1] for point in points]

        try:
            z_values = [point[2] for point in points]
        except IndexError:
            z_values = [0]*len(points)

        return x_values, y_values, z_values


class ScatterMatplotlib(Plot):

    def __init__(self, plot_title: str, number_of_objectives: int):
        """ Creates a new :class:`ScatterPlot` instance. Suitable for problems with 2 or 3 objectives.

        :param plot_title: Title of the scatter diagram.
        :param number_of_objectives: Number of objectives to be used (2D/3D).
        """
        super().__init__(plot_title, number_of_objectives)

        # Initialize a plot
        self.fig = plt.figure()
        self.sc = None
        self.axis = None

        self.__initialize()

    def __initialize(self) -> None:
        """ Initialize the scatter plot for the first time. """
        logger.info("Generating plot...")

        # Initialize a plot
        self.fig.canvas.set_window_title('jMetalPy')

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

        logger.info("Plot initialized")

    def __plot(self, x_values, y_values, z_values, color: str = '#98FB98', marker: str = 'o', msize: int = 3):
        if self.number_of_objectives == 2:
            self.sc, = self.axis.plot(x_values, y_values,
                                      color=color, marker=marker, markersize=msize, ls='None', picker=10)
        else:
            self.sc, = self.axis.plot(x_values, y_values, z_values,
                                      color=color, marker=marker, markersize=msize, ls='None', picker=10)

    def plot(self, solution_list: List[S], reference: List[S], output: str='', show: bool=True) -> None:
        if reference:
            logger.info("Reference front found")
            ref_x_values, ref_y_values, ref_z_values = self.get_objectives(reference)
            self.__plot(ref_x_values, ref_y_values, ref_z_values, color='#323232', marker='*')

        x_values, y_values, z_values = self.get_objectives(solution_list)
        self.__plot(x_values, y_values, z_values)

        if output:
            self.__save(output)
        if show:
            self.fig.canvas.mpl_connect('pick_event', lambda event: self.__pick_handler(event, solution_list))
            plt.show()

    def update(self, solution_list: List[S], reference: List[S], new_title: str='', persistence: bool=True) -> None:
        if self.sc is None:
            logger.warning("Plot is none! Generating first plot...")
            self.plot(solution_list, reference, show=False)

        x_values, y_values, z_values = self.get_objectives(solution_list)

        if persistence:
            # Replace with new points
            self.sc.set_data(x_values, y_values)

            if self.number_of_objectives == 3:
                self.sc.set_3d_properties(z_values)
        else:
            # Add new points
            self.__plot(x_values, y_values, z_values)

        # Also, add event handler
        event_handler = \
            self.fig.canvas.mpl_connect('pick_event', lambda event: self.__pick_handler(event, solution_list))

        # Update title with new times and evaluations
        self.fig.suptitle(new_title, fontsize=13)

        # Re-align the axis
        self.axis.relim()
        self.axis.autoscale_view(True, True, True)

        # Draw
        self.fig.canvas.draw()
        plt.pause(0.01)

        # Disconnect the pick event for the next update
        self.fig.canvas.mpl_disconnect(event_handler)

    def __save(self, file_name: str, fmt: str = 'png', dpi: int = 200):
        supported_formats = ["eps", "jpeg", "jpg", "pdf", "pgf", "png", "ps",
                             "raw", "rgba", "svg", "svgz", "tif", "tiff"]

        if fmt not in supported_formats:
            raise Exception("{0} is not a valid format! Use one of these instead: {0}".format(fmt, supported_formats))

        self.fig.savefig(file_name + '.' + fmt, format=fmt, dpi=dpi)

    def __retrieve_info(self, x_val: float, y_val: float, solution: Solution) -> None:
        logger.info("Output file: " + '{0}-{1}'.format(x_val, y_val))

        with open('{0}-{1}'.format(x_val, y_val), 'w') as of:
            of.write(solution.__str__())

    def __pick_handler(self, event, solution_list: List[S]):
        """ Handler for picking points from the plot. """
        line, ind = event.artist, event.ind[0]
        x, y = line.get_xdata(), line.get_ydata()

        logger.debug('Selected resources point ({0}): ({1}, {2})'.format(ind, x[ind], y[ind]))

        sol = next((solution for solution in solution_list
                    if solution.objectives[0] == x[ind] and solution.objectives[1] == y[ind]), None)

        if sol is not None:
            self.__retrieve_info(x[ind], y[ind], sol)
        else:
            logger.warning("Solution is none.")
            return True


class ScatterBokeh(Plot):

    def __init__(self, plot_title: str, number_of_objectives: int, ws_url: str='localhost:5006'):
        super().__init__(plot_title, number_of_objectives)

        if self.number_of_objectives == 2:
            self.source = ColumnDataSource(data=dict(x=[], y=[], str=[]))
        elif self.number_of_objectives == 3:
            self.source = ColumnDataSource(data=dict(x=[], y=[], z=[], str=[]))
        else:
            raise Exception('Wrong number of objectives: {0}'.format(number_of_objectives))

        self.client = ClientSession(websocket_url="ws://{0}/ws".format(ws_url))
        self.doc = curdoc()
        self.doc.title = plot_title
        self.figure_xy = None
        self.figure_xz = None
        self.figure_yz = None

        self.__initialize()

    def __initialize(self) -> None:
        """ Set-up tools for plot. """
        code = '''
            selected = source.selected['1d']['indices'][0]
            var str = source.resources.str[selected]
            alert(str)
        '''

        callback = CustomJS(args=dict(source=self.source), code=code)
        self.plot_tools = [TapTool(callback=callback), WheelZoomTool(), 'save', 'pan',
                           HoverTool(tooltips=[("index", "$index"), ("(x,y)", "($x, $y)")])]

    def plot(self, solution_list: List[S], reference: List[S]=None, output: str='', show: bool=True) -> None:
        # This is important to purge resources (if any) between calls
        reset_output()

        # Set up figure
        self.figure_xy = Figure(output_backend='webgl', sizing_mode='scale_width', title=self.plot_title, tools=self.plot_tools)
        self.figure_xy.scatter(x='x', y='y', legend='solution', fill_alpha=0.7, source=self.source)
        self.figure_xy.xaxis.axis_label = self.xaxis_label
        self.figure_xy.yaxis.axis_label = self.yaxis_label

        x_values, y_values, z_values = self.get_objectives(solution_list)

        if self.number_of_objectives == 2:
            # Plot reference solution list (if any)
            if reference:
                ref_x_values, ref_y_values, _ = self.get_objectives(reference)
                self.figure_xy.line(x=ref_x_values, y=ref_y_values, legend='reference', color='green')

            # Push resources to server
            self.source.stream({'x': x_values, 'y': y_values, 'str': [s.__str__() for s in solution_list]})
            self.doc.add_root(column(self.figure_xy))
        else:
            # Add new figures for each axis
            self.figure_xz = Figure(title='xz', output_backend='webgl', sizing_mode='scale_width', tools=self.plot_tools)
            self.figure_xz.scatter(x='x', y='z', legend='solution', fill_alpha=0.7, source=self.source)
            self.figure_xz.xaxis.axis_label = self.xaxis_label
            self.figure_xz.yaxis.axis_label = self.zaxis_label

            self.figure_yz = Figure(title='yz', output_backend='webgl', sizing_mode='scale_width', tools=self.plot_tools)
            self.figure_yz.scatter(x='y', y='z', legend='solution', fill_alpha=0.7, source=self.source)
            self.figure_yz.xaxis.axis_label = self.yaxis_label
            self.figure_yz.yaxis.axis_label = self.zaxis_label

            # Plot reference solution list (if any)
            if reference:
                ref_x_values, ref_y_values, ref_z_values = self.get_objectives(reference)
                self.figure_xy.line(x=ref_x_values, y=ref_y_values, legend='reference', color='green')
                self.figure_xz.line(x=ref_x_values, y=ref_z_values, legend='reference', color='green')
                self.figure_yz.line(x=ref_y_values, y=ref_z_values, legend='reference', color='green')

            # Push resources to server
            self.source.stream({'x': x_values, 'y': y_values, 'z': z_values, 'str': [s.__str__() for s in solution_list]})
            self.doc.add_root(row(self.figure_xy, self.figure_xz, self.figure_yz))

        self.client.push(self.doc)

        if output:
            self.__save(output)
        if show:
            self.client.show()

    def update(self, solution_list: List[S], reference: List[S], new_title: str='', persistence: bool=False) -> None:
        # Check if plot has not been initialized first
        if self.figure_xy is None:
            self.plot(solution_list, reference)

        if not persistence:
            rollover = len(solution_list)
        else:
            rollover = None

        self.figure_xy.title.text = new_title
        x_values, y_values, z_values = self.get_objectives(solution_list)

        if self.number_of_objectives == 2:
            self.source.stream({'x': x_values, 'y': y_values, 'str': [s.__str__() for s in solution_list]},
                               rollover=rollover)
        else:
            self.source.stream({'x': x_values, 'y': y_values, 'z': z_values, 'str': [s.__str__() for s in solution_list]},
                               rollover=rollover)

    def __save(self, file_name: str):
        # env = Environment(loader=FileSystemLoader(BASE_PATH + '/util/'))
        # env.filters['json'] = lambda obj: Markup(json.dumps(obj))

        html = file_html(models=self.doc, resources=CDN)
        with open(file_name + '.html', 'w') as of:
            of.write(html)

    def disconnect(self):
        if self.is_connected():
            self.client.close()

    def is_connected(self) -> bool:
        return self.client.connected
