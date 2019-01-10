import logging
from typing import TypeVar, List

import holoviews as hv
import matplotlib.pyplot as plt
from IPython.display import display
from holoviews.streams import Pipe
from mpl_toolkits.mplot3d import Axes3D

from jmetal.util.visualization.plotting import Plot

LOGGER = logging.getLogger('jmetal')

hv.extension('matplotlib')

S = TypeVar('S')

"""
.. module:: streaming
   :platform: Unix, Windows
   :synopsis: Classes for plotting solutions in real-time.

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""

# Define some colors
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)


class StreamingPlot(Plot):

    def __init__(self,
                 plot_title: str,
                 reference_front: List[S] = None,
                 reference_point: list = None,
                 axis_labels: list = None):
        super(StreamingPlot, self).__init__(plot_title, reference_front, reference_point, axis_labels)

        import warnings
        warnings.filterwarnings("ignore", ".*GUI is implemented.*")

        self.fig, self.ax = plt.subplots()
        self.sc = None
        self.axis = None

    def plot(self, front: List[S]) -> None:
        # Get data
        points, dimension = self.get_points(front)

        # Create an empty figure
        self.create_layout(dimension)

        # If any reference point, plot
        if self.reference_point:
            self.sc, = self.ax.plot(*[[point] for point in self.reference_point],
                                    c=tableau20[10], ls='None', marker='*', markersize=3)

        # If any reference front, plot
        if self.reference_front:
            rpoints, _ = self.get_points(self.reference_front)
            self.sc, = self.ax.plot(*[rpoints[column].tolist() for column in rpoints.columns.values],
                                    c=tableau20[15], ls='None', marker='*', markersize=3)

        # Plot data
        self.sc, = self.ax.plot(*[points[column].tolist() for column in points.columns.values],
                                ls='None', marker='o', markersize=4)

    def update(self, front: List[S]) -> None:
        if self.sc is None:
            raise Exception('Figure is none')

        points, dimension = self.get_points(front)

        # Replace with new points
        self.sc.set_data(points[0], points[1])

        if dimension == 3:
            self.sc.set_3d_properties(points[2])

        # Re-align the axis
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)

        try:
            self.fig.canvas.draw()
        except KeyboardInterrupt:
            pass

        plt.pause(0.01)

    def create_layout(self, dimension: int) -> None:
        self.fig.canvas.set_window_title(self.plot_title)

        if dimension == 2:
            # Stylize axis
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.ax.get_xaxis().tick_bottom()
            self.ax.get_yaxis().tick_left()
        elif dimension == 3:
            self.ax = Axes3D(self.fig)
            self.ax.autoscale(enable=True, axis='both')
        else:
            raise Exception('Dimension must be either 2 or 3')

        self.ax.set_autoscale_on(True)
        self.ax.autoscale_view(True, True, True)

        # Style options
        self.ax.grid(color='#f0f0f5', linestyle='-', linewidth=0.5, alpha=0.5)
        self.ax.set_title(self.plot_title)


class IStreamingPlot(Plot):

    def __init__(self,
                 plot_title: str,
                 reference_front: List[S] = None,
                 reference_point: list = None,
                 axis_labels: list = None):
        super(IStreamingPlot, self).__init__(plot_title, reference_front, reference_point, axis_labels)
        self.figure = None
        self.pipe = Pipe(data=[])

    def plot(self, solutions: List[S]):
        # Get data
        points, dimension = self.get_points(solutions)
        points = points.values.tolist()

        # Create an empty figure
        self.create_layout(dimension)

        # If any reference point, plot
        if self.reference_point:
            if dimension == 2:
                self.figure = self.figure * hv.Scatter(self.reference_point, label='Reference point')
            elif dimension == 3:
                self.figure = self.figure * hv.Scatter3D(self.reference_point, label='Reference point')

        # If any reference front, plot
        if self.reference_front:
            rpoints, dimension = self.get_points(self.reference_front)
            rpoints = rpoints.values.tolist()

            if dimension == 2:
                self.figure = self.figure * hv.Scatter(rpoints, label='Reference front')
            elif dimension == 3:
                self.figure = self.figure * hv.Scatter3D(rpoints, label='Reference front')

        # Plot data
        display(self.figure)  # Display figure in IPython
        self.pipe.send(points)

    def update(self, solutions: List[S]):
        if self.figure is None:
            raise Exception('Figure is none')

        points, _ = self.get_points(solutions)
        points = points.values.tolist()

        self.pipe.send(points)

    def create_layout(self, dimension: int):
        if dimension == 2:
            self.figure = hv.DynamicMap(hv.Scatter, streams=[self.pipe])
        elif dimension == 3:
            self.figure = hv.DynamicMap(hv.Scatter3D, streams=[self.pipe])
        else:
            raise Exception('Dimension must be either 2 or 3')

    def export(self, file_name: str, file_format: str = 'svg'):
        renderer = hv.renderer('matplotlib').instance(fig=file_format)
        renderer.save(self.figure, file_name)
