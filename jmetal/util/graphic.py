import logging
from abc import ABCMeta
from string import Template
from typing import TypeVar, List

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotly import graph_objs as go
from plotly.offline import plot
from pandas import DataFrame, merge

jMetalPyLogger = logging.getLogger('jMetalPy')
S = TypeVar('S')

"""
.. module:: visualization
   :platform: Unix, Windows
   :synopsis: Classes for plotting fronts.

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Plot:

    __metaclass__ = ABCMeta

    def __init__(self, plot_title: str, axis_labels: list):
        self.plot_title = plot_title
        self.axis_labels = axis_labels

        self.number_of_objectives: int = None

    @staticmethod
    def get_objectives(front: List[S]) -> DataFrame:
        """ Get objectives for each solution of the front.

        :param front: List of solutions.
        :return: Pandas dataframe with one column for each objective and one row for each solution. """
        if front is None:
            raise Exception('Front is none!')

        return DataFrame(list(solution.objectives for solution in front))


class ScatterStreaming(Plot):

    def __init__(self, plot_title: str, axis_labels: list = None):
        """ Creates a new :class:`ScatterStreaming` instance. Suitable for problems with 2 or 3 objectives in streaming.

        :param plot_title: Title of the diagram.
        :param axis_labels: List of axis labels. """
        super(ScatterStreaming, self).__init__(plot_title, axis_labels)

        import warnings
        warnings.filterwarnings("ignore", ".*GUI is implemented.*")

        self.fig = plt.figure()
        self.sc = None
        self.axis = None

    def plot(self, front: List[S], reference_front: List[S], filename: str = '', show: bool = True) -> None:
        """ Plot a front of solutions (2D or 3D).

        :param front: List of solutions.
        :param reference_front: Reference solution list (if any).
        :param filename: If specified, save the plot into a file.
        :param show: If True, show the final diagram (default to True). """
        objectives = self.get_objectives(front)

        # Initialize plot
        self.number_of_objectives = objectives.shape[1]
        self.__initialize()

        if reference_front:
            jMetalPyLogger.info('Reference front found')
            ref_objectives = self.get_objectives(reference_front)

            if self.number_of_objectives == 2:
                self.__plot(ref_objectives[0], ref_objectives[1], None,
                            color='#323232', marker='*', markersize=3)
            else:
                self.__plot(ref_objectives[0], ref_objectives[1], ref_objectives[2],
                            color='#323232', marker='*', markersize=3)

        if self.number_of_objectives == 2:
            self.__plot(objectives[0], objectives[1], None, color='#98FB98', marker='o', markersize=3)
        else:
            self.__plot(objectives[0], objectives[1], objectives[2], color='#98FB98', marker='o', markersize=3)

        if filename:
            self.fig.savefig(filename, format='png', dpi=200)
        if show:
            self.fig.canvas.mpl_connect('pick_event', lambda event: self.__pick_handler(front, event))
            plt.show()

    def update(self, front: List[S], reference_front: List[S], rename_title: str = '',
               persistence: bool = True) -> None:
        """ Update an already created plot.

        :param front: List of solutions.
        :param reference_front: Reference solution list (if any).
        :param rename_title: New title of the plot.
        :param persistence: If True, keep old points; else, replace them with new values.
        """
        if self.sc is None:
            jMetalPyLogger.warning('Plot must be initialized first.')
            self.plot(front, reference_front, show=False)
            return

        objectives = self.get_objectives(front)

        if persistence:
            # Replace with new points
            self.sc.set_data(objectives[0], objectives[1])

            if self.number_of_objectives == 3:
                self.sc.set_3d_properties(objectives[2])
        else:
            # Add new points
            if self.number_of_objectives == 2:
                self.__plot(objectives[0], objectives[1], None, color='#98FB98', marker='o', markersize=3)
            else:
                self.__plot(objectives[0], objectives[1], objectives[2], color='#98FB98', marker='o', markersize=3)

        # Also, add event handler
        event_handler = \
            self.fig.canvas.mpl_connect('pick_event', lambda event: self.__pick_handler(front, event))

        # Update title with new times and evaluations
        self.fig.suptitle(rename_title, fontsize=13)

        # Re-align the axis
        self.axis.relim()
        self.axis.autoscale_view(True, True, True)

        try:
            # Draw
            self.fig.canvas.draw()
        except KeyboardInterrupt:
            pass

        plt.pause(0.01)

        # Disconnect the pick event for the next update
        self.fig.canvas.mpl_disconnect(event_handler)

    def __initialize(self) -> None:
        """ Initialize the scatter plot for the first time. """
        jMetalPyLogger.info('Generating plot')

        # Initialize a plot
        self.fig.canvas.set_window_title('jMetalPy')

        if self.number_of_objectives == 2:
            self.axis = self.fig.add_subplot(111)

            # Stylize axis
            self.axis.spines['top'].set_visible(False)
            self.axis.spines['right'].set_visible(False)
            self.axis.get_xaxis().tick_bottom()
            self.axis.get_yaxis().tick_left()
        elif self.number_of_objectives == 3:
            self.axis = Axes3D(self.fig)
            self.axis.autoscale(enable=True, axis='both')
        else:
            raise Exception('Number of objectives must be either 2 or 3')

        self.axis.set_autoscale_on(True)
        self.axis.autoscale_view(True, True, True)

        # Style options
        self.axis.grid(color='#f0f0f5', linestyle='-', linewidth=1, alpha=0.5)
        self.fig.suptitle(self.plot_title, fontsize=13)

        jMetalPyLogger.info('Plot initialized')

    def __plot(self, x_values, y_values, z_values, **kwargs) -> None:
        if self.number_of_objectives == 2:
            self.sc, = self.axis.plot(x_values, y_values, ls='None', picker=10, **kwargs)
        else:
            self.sc, = self.axis.plot(x_values, y_values, z_values, ls='None', picker=10, **kwargs)

    def __pick_handler(self, front: List[S], event):
        """ Handler for picking points from the plot. """
        line, ind = event.artist, event.ind[0]
        x, y = line.get_xdata(), line.get_ydata()

        jMetalPyLogger.debug('Selected front point ({0}): ({1}, {2})'.format(ind, x[ind], y[ind]))

        sol = next((solution for solution in front
                    if solution.objectives[0] == x[ind] and solution.objectives[1] == y[ind]), None)

        if sol is not None:
            with open('{0}-{1}'.format(x[ind], y[ind]), 'w') as of:
                of.write(sol.__str__())
        else:
            jMetalPyLogger.warning('Solution is none')
            return True


class FrontPlot(Plot):

    def __init__(self, plot_title: str, axis_labels: list = None):
        """ Creates a new :class:`FrontPlot` instance. Suitable for problems with 2 or more objectives.

        :param plot_title: Title of the graph.
        :param axis_labels: List of axis labels. """
        super(FrontPlot, self).__init__(plot_title, axis_labels)

        self.figure: go.Figure = None
        self.layout = None
        self.data = []

    def plot(self, front: List[S], reference_front: List[S] = None, normalize: bool = False) -> None:
        """ Plot a front of solutions (2D, 3D or parallel coordinates).

        :param front: List of solutions.
        :param reference_front: Reference solution list (if any).
        :param normalize: Normalize the input front between 0 and 1 (for problems with more than 3 objectives). """
        self.__initialize()

        if reference_front:
            objectives = self.get_objectives(reference_front)
            trace = self.__generate_trace(objectives=objectives, legend='reference front', normalize=normalize,
                                          color='rgb(2, 130, 242)')
            self.data.append(trace)

        objectives = self.get_objectives(front)
        metadata = list(solution.__str__() for solution in front)
        trace = self.__generate_trace(objectives=objectives, metadata=metadata, legend='front', normalize=normalize,
                                      symbol='diamond-open')
        self.data.append(trace)

        self.figure = go.Figure(data=self.data, layout=self.layout)

    def update(self, data: List[S], normalize: bool = False, legend: str = '') -> None:
        """ Update an already created graph with new data.

        :param data: List of solutions to be included.
        :param legend: Legend to be included.
        :param normalize: Normalize the input front between 0 and 1 (for problems with more than 3 objectives). """
        if self.figure is None:
            jMetalPyLogger.warning('Plot must be initialized first.')
            self.plot(data, reference_front=None, normalize=normalize)
            return

        objectives = self.get_objectives(data)
        new_data = self.__generate_trace(objectives=objectives, legend=legend, normalize=normalize,
                                         color='rgb(255, 170, 0)')
        self.data.append(new_data)

        self.figure = go.Figure(data=self.data, layout=self.layout)

    def to_html(self, filename: str='front') -> str:
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
                ''' + self.export(include_plotlyjs=False) + '''
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
                </script>
            </body>
        </html>'''

        with open(filename + '.html', 'w') as outf:
            outf.write(html_string)

        return html_string

    def export(self, filename: str = '', include_plotlyjs: bool = False) -> str:
        """ Export as a `div` for embedding the graph in an HTML file.

        :param filename: Output file name (if desired, default to None).
        :param include_plotlyjs: If True, include plot.ly JS script (default to False).
        :return: Script as string. """
        script = plot(self.figure, output_type='div', include_plotlyjs=include_plotlyjs, show_link=False)

        if filename:
            with open(filename + '.html', 'w') as outf:
                outf.write(script)

        return script

    def __initialize(self):
        """ Initialize the graph for the first time. """
        jMetalPyLogger.info('Generating graph')

        self.layout = go.Layout(
            margin=dict(l=80, r=80, b=80, t=150),
            title=self.plot_title,
            scene=dict(
                xaxis=dict(title=self.axis_labels[0:1][0] if self.axis_labels[0:1] else None),
                yaxis=dict(title=self.axis_labels[1:2][0] if self.axis_labels[1:2] else None),
                zaxis=dict(title=self.axis_labels[2:3][0] if self.axis_labels[2:3] else None)
            ),
            images=[dict(
                source='https://raw.githubusercontent.com/jMetal/jMetalPy/master/docs/source/jmetalpy.png',
                xref='paper', yref='paper',
                x=0, y=1.05,
                sizex=0.1, sizey=0.1,
                xanchor="left", yanchor="bottom"
            )],
            hovermode='closest'
        )

    def __generate_trace(self, objectives: DataFrame, metadata: list = None, legend: str = '', normalize: bool = False,
                         **kwargs):
        number_of_objectives = objectives.shape[1]

        if normalize:
            objectives = (objectives - objectives.min()) / (objectives.max() - objectives.min())

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

        if number_of_objectives == 2:
            trace = go.Scattergl(
                x=objectives[0],
                y=objectives[1],
                mode='markers',
                marker=marker,
                name=legend,
                customdata=metadata
            )
        elif number_of_objectives == 3:
            trace = go.Scatter3d(
                x=objectives[0],
                y=objectives[1],
                z=objectives[2],
                mode='markers',
                marker=marker,
                name=legend,
                customdata=metadata
            )
        else:
            dimensions = list()
            for column in objectives:
                dimensions.append(
                    dict(range=[0, 1],
                         label=self.axis_labels[column:column+1][0] if self.axis_labels[column:column+1] else None,
                         values=objectives[column])
                )

            trace = go.Parcoords(
                line=dict(color='blue'),
                dimensions=dimensions,
                name=legend,
            )

        return trace

    def __save(self, filename: str = 'front', show: bool = False) -> None:
        """ Save the graph. """
        plot(self.figure, filename=filename + '.html', auto_open=show, show_link=False)
