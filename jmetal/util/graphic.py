import logging
from abc import ABC
from typing import TypeVar, List, Tuple

import holoviews as hv
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from holoviews.streams import Pipe
from mpl_toolkits.mplot3d import Axes3D
from plotly import graph_objs as go
from plotly.offline import plot

LOGGER = logging.getLogger('jmetal')

hv.extension('matplotlib')

S = TypeVar('S')

"""
.. module:: visualization
   :platform: Unix, Windows
   :synopsis: Classes for plotting solutions.

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


class Plot(ABC):

    def __init__(self, plot_title: str, reference_front: List[S], reference_point: list, axis_labels: list):
        """
        :param plot_title: Title of the graph.
        :param axis_labels: List of axis labels.
        :param reference_point:
        :param reference_front:
        """
        self.plot_title = plot_title
        self.axis_labels = axis_labels
        self.reference_point = reference_point
        self.reference_front = reference_front
        self.dimension = None

    @staticmethod
    def get_points(front: List[S]) -> Tuple[pd.DataFrame, int]:
        """ Get points for each solution of the front.

        :param front: List of solutions.
        :return: Pandas dataframe with one column for each objective and one row for each solution. """
        if front is None:
            raise Exception('Front is none!')

        points = pd.DataFrame(list(solution.objectives for solution in front))
        return points, points.shape[1]

    def two_dim(self, solutions: S):
        points, _ = self.get_points(solutions)
        points.plot(kind='scatter', x=0, y=1)

    def three_dim(self, solutions: S):
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter([s.objectives[0] for s in solutions],
                   [s.objectives[1] for s in solutions],
                   [s.objectives[2] for s in solutions])
        ax.set_xlim([0, 1.1])
        ax.set_ylim([0, 1.1])
        ax.set_zlim([0, 1.1])
        ax.view_init(elev=30.0, azim=15)
        ax.set_title(self.plot_title)

        plt.show()


class StreamingPlot(Plot):

    def __init__(self, plot_title: str, reference_front: List[S] = None, reference_point: list = None,
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
                                    c=tableau20[10], ls='None', marker='*', markersize=2)

        # If any reference front, plot
        if self.reference_front:
            rpoints, _ = self.get_points(self.reference_front)
            self.sc, = self.ax.plot(*[rpoints[column].tolist() for column in rpoints.columns.values],
                                    c=tableau20[15], ls='None', marker='*', markersize=2)

        # Plot data
        self.sc, = self.ax.plot(*[points[column].tolist() for column in points.columns.values],
                                c=tableau20[3], ls='None', marker='o', markersize=2)

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
            raise Exception('Number of objectives must be either 2 or 3')

        self.ax.set_autoscale_on(True)
        self.ax.autoscale_view(True, True, True)

        # Style options
        self.ax.grid(color='#f0f0f5', linestyle='-', linewidth=0.5, alpha=0.5)
        self.fig.suptitle(self.plot_title, fontsize=13)


class IStreamingPlot(Plot):

    def __init__(self, plot_title: str, reference_front: List[S] = None, reference_point: list = None,
                 axis_labels: list = None):
        super(IStreamingPlot, self).__init__(plot_title, reference_front, reference_point, axis_labels)
        self.figure = None
        self.pipe = Pipe(data=[])

    def plot(self, front: List[S]):
        # Get data
        points, dimension = self.get_points(front)
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

    def update(self, front: List[S]):
        if self.figure is None:
            raise Exception('Figure is none')

        points, _ = self.get_points(front)
        points = points.values.tolist()

        self.pipe.send(points)

    def create_layout(self, dimension: int):
        if dimension == 2:
            self.figure = hv.DynamicMap(hv.Scatter, streams=[self.pipe])
        elif dimension == 3:
            self.figure = hv.DynamicMap(hv.Scatter3D, streams=[self.pipe])
        else:
            raise Exception('Number of objectives must be either 2 or 3')

    def export(self, file_name: str, file_format: str = 'svg'):
        renderer = hv.renderer('matplotlib').instance(fig=file_format)
        renderer.save(self.figure, file_name)


class InteractivePlot(Plot):

    def __init__(self, plot_title: str, reference_front: List[S] = None, reference_point: list = None,
                 axis_labels: list = None):
        super(InteractivePlot, self).__init__(plot_title, reference_front, reference_point, axis_labels)
        self.figure = None
        self.layout = None
        self.data = []

    def plot(self, front: List[S], normalize: bool = False) -> None:
        """ Plot a front of solutions (2D, 3D or parallel coordinates).

        :param front: List of solutions.
        :param normalize: Normalize the input front between 0 and 1 (for problems with more than 3 objectives).
        """
        self.layout = go.Layout(
            margin=dict(l=80, r=80, b=80, t=150),
            height=800,
            title=self.plot_title,
            scene=dict(
                xaxis=dict(title=self.axis_labels[0:1][0] if self.axis_labels[0:1] else None),
                yaxis=dict(title=self.axis_labels[1:2][0] if self.axis_labels[1:2] else None),
                zaxis=dict(title=self.axis_labels[2:3][0] if self.axis_labels[2:3] else None)
            ),
            hovermode='closest'
        )

        # If any reference front, plot
        if self.reference_front:
            points, _ = self.get_points(self.reference_front)
            trace = self.__generate_trace(points=points, legend='Reference front', normalize=normalize,
                                          color='rgb(2, 130, 242)')
            self.data.append(trace)

        # Get points and metadata
        points, _ = self.get_points(front)
        metadata = list(solution.__str__() for solution in front)

        trace = self.__generate_trace(points=points, metadata=metadata, legend='Front', normalize=normalize,
                                      symbol='diamond-open')
        self.data.append(trace)

        # Plot the figure
        self.figure = go.Figure(data=self.data, layout=self.layout)

    def update(self, points: List[S], normalize: bool = False, legend: str = '') -> None:
        if self.figure is None:
            raise Exception('Figure is none')

        points, _ = self.get_points(points)
        new_data = self.__generate_trace(points=points, legend=legend, normalize=normalize, color='rgb(255, 170, 0)')

        self.data.append(new_data)
        self.figure = go.Figure(data=self.data, layout=self.layout)

    def export_to_html(self, filename: str = 'front') -> str:
        """ Export the graph to an interactive HTML (solutions can be selected to show some metadata).

        :param filename: Output file name.
        :return: Script as string. """
        html_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                <meta charset="utf-8"/>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <script src="https://unpkg.com/sweetalert2@7.7.0/dist/sweetalert2.all.js"></script>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
            </head>
            <body>
                ''' + self.export_to_div(filename=None, include_plotlyjs=False) + '''
                <script>                
                    var myPlot = document.querySelectorAll('div')[0];
                    myPlot.on('plotly_click', function(data){
                        var pts = '';

                        for(var i=0; i < data.points.length; i++){
                            pts = '(x, y) = ('+data.points[i].x +', '+ data.points[i].y.toPrecision(4)+')';
                            cs = data.points[i].customdata
                        }

                        if(typeof cs !== "undefined"){
                            swal({
                              title: 'Closest solution clicked:',
                              text: cs,
                              type: 'info',
                              position: 'bottom-end'
                            })
                        }
                    });

                    window.onresize = function() {
                       Plotly.Plots.resize(myPlot);
                    };
                </script>
            </body>
        </html>'''

        with open(filename + '.html', 'w') as outf:
            outf.write(html_string)

        return html_string

    def export_to_div(self, filename=None, include_plotlyjs: bool = False) -> str:
        """ Export as a `div` for embedding the graph in an HTML file.

        :param filename: Output file name (if desired, default to None).
        :param include_plotlyjs: If True, include plot.ly JS script (default to False).
        :return: Script as string.
        """
        script = plot(self.figure, output_type='div', include_plotlyjs=include_plotlyjs, show_link=False)

        if filename:
            with open(filename + '.html', 'w') as outf:
                outf.write(script)

        return script

    def __generate_trace(self, points: pd.DataFrame, legend: str, metadata: list = None, normalize: bool = False,
                         **kwargs):
        dimension = points.shape[1]

        if normalize:
            points = (points - points.min()) / (points.max() - points.min())

        marker = dict(
            color='rgb(127, 127, 127)',
            size=3,
            symbol='x',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.8
        )
        marker.update(**kwargs)

        if dimension == 2:
            trace = go.Scattergl(
                x=points[0],
                y=points[1],
                mode='markers',
                marker=marker,
                name=legend,
                customdata=metadata
            )
        elif dimension == 3:
            trace = go.Scatter3d(
                x=points[0],
                y=points[1],
                z=points[2],
                mode='markers',
                marker=marker,
                name=legend,
                customdata=metadata
            )
        else:
            dimensions = list()
            for column in points:
                dimensions.append(
                    dict(range=[0, 1],
                         label=self.axis_labels[column:column + 1][0] if self.axis_labels[column:column + 1] else None,
                         values=points[column])
                )

            trace = go.Parcoords(
                line=dict(color='blue'),
                dimensions=dimensions,
                name=legend,
            )

        return trace
