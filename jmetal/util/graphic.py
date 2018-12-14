import logging
from abc import ABC
from typing import TypeVar, List

import holoviews as hv
from IPython.display import display
from holoviews.streams import Pipe
from pandas import DataFrame
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


class Plot(ABC):

    def __init__(self, plot_title: str, axis_labels: list):
        self.plot_title = plot_title
        self.axis_labels = axis_labels

        self.number_of_objectives = None

    @staticmethod
    def get_objectives(front: List[S]) -> DataFrame:
        """ Get objectives for each solution of the front.

        :param front: List of solutions.
        :return: Pandas dataframe with one column for each objective and one row for each solution. """
        if front is None:
            raise Exception('Front is none!')

        return DataFrame(list(solution.objectives for solution in front))


class StreamingPlot(Plot):

    def __init__(self, plot_title: str = 'jmetal', axis_labels: list = None):
        super(StreamingPlot, self).__init__(plot_title, axis_labels)
        self.figure = None
        self.pipe = Pipe(data=[])

    def plot(self, front: List[S], reference_front: List[S] = None):
        objectives = self.get_objectives(front)
        dimension = objectives.shape[1]

        if reference_front:
            reference_objectives = self.get_objectives(reference_front).values.tolist()
            self.figure = hv.Scatter(reference_objectives, label='Reference front') * hv.DynamicMap(
                hv.Scatter if dimension == 2 else hv.Scatter3D, streams=[self.pipe])
        else:
            self.figure = hv.DynamicMap(hv.Scatter if dimension == 2 else hv.Scatter3D, streams=[self.pipe])

        display(self.figure)
        self.pipe.send(objectives.values.tolist())

    def update(self, front: List[S], reference_front: List[S] = None):
        if self.figure is None:
            self.plot(front, reference_front)
            return

        objectives = self.get_objectives(front)
        self.pipe.send(objectives.values.tolist())

    def export(self, file_name: str, file_format: str = 'svg'):
        renderer = hv.renderer('matplotlib').instance(fig=file_format)
        renderer.save(self.figure, file_name)

    def show(self):
        return self.figure


class InteractivePlot(Plot):

    def __init__(self, plot_title: str = 'jmetal', axis_labels: list = None):
        """ Creates a new :class:`FrontPlot` instance. Suitable for problems with 2 or more objectives.

        :param plot_title: Title of the graph.
        :param axis_labels: List of axis labels. """
        super(InteractivePlot, self).__init__(plot_title, axis_labels)
        self.figure = None
        self.layout = None
        self.data = []

    def plot(self, front: List[S], reference_front: List[S] = None, normalize: bool = False) -> None:
        """ Plot a front of solutions (2D, 3D or parallel coordinates).

        :param front: List of solutions.
        :param reference_front: Reference solution list (if any).
        :param normalize: Normalize the input front between 0 and 1 (for problems with more than 3 objectives).
        """
        self.create_layout()

        if reference_front:
            objectives = self.get_objectives(reference_front)
            trace = self.generate_trace(objectives=objectives, legend='reference front', normalize=normalize,
                                        color='rgb(2, 130, 242)')
            self.data.append(trace)

        objectives = self.get_objectives(front)
        metadata = list(solution.__str__() for solution in front)
        trace = self.generate_trace(objectives=objectives, metadata=metadata, legend='front', normalize=normalize,
                                    symbol='diamond-open')
        self.data.append(trace)

        self.figure = go.Figure(data=self.data, layout=self.layout)

    def update(self, data: List[S], normalize: bool = False, legend: str = '') -> None:
        """ Update an already created graph with new data.

        :param data: List of solutions to be included.
        :param legend: Legend to be included.
        :param normalize: Normalize the input front between 0 and 1 (for problems with more than 3 objectives).
        """
        if self.figure is None:
            self.plot(data, reference_front=None, normalize=normalize)
            return

        objectives = self.get_objectives(data)
        new_data = self.generate_trace(objectives=objectives, legend=legend, normalize=normalize,
                                       color='rgb(255, 170, 0)')
        self.data.append(new_data)

        self.figure = go.Figure(data=self.data, layout=self.layout)

    def export_html(self, filename: str = 'front') -> str:
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
                <style>
                    .float{
                      position:fixed;
                      right:40px;
                      bottom:40px;
                    }
                </style>
            </head>
            <body>
                <a class="float" href="https://jmetalpy.readthedocs.io/en/latest/">
                  <img src="https://raw.githubusercontent.com/jMetal/jMetalPy/master/docs/source/jmetalpy.png" height="20px"/>
                </a>
                ''' + self.export_div(include_plotlyjs=False) + '''
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

    def export_div(self, filename: str = 'output', include_plotlyjs: bool = False) -> str:
        """ Export as a `div` for embedding the graph in an HTML file.

        :param filename: Output file name (if desired, default to None).
        :param include_plotlyjs: If True, include plot.ly JS script (default to False).
        :return: Script as string.
        """
        script = plot(self.figure, output_type='div', include_plotlyjs=include_plotlyjs, show_link=False)

        with open(filename + '.html', 'w') as outf:
            outf.write(script)

        return script

    def create_layout(self):
        """ Initialize the graph for the first time.
        """
        LOGGER.debug('Generating graph')

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

    def generate_trace(self, objectives: DataFrame, metadata: list = None, legend: str = '', normalize: bool = False,
                       **kwargs):
        dimensions = objectives.shape[1]

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

        if dimensions == 2:
            trace = go.Scattergl(
                x=objectives[0],
                y=objectives[1],
                mode='markers',
                marker=marker,
                name=legend,
                customdata=metadata
            )
        elif dimensions == 3:
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
                         label=self.axis_labels[column:column + 1][0] if self.axis_labels[column:column + 1] else None,
                         values=objectives[column])
                )

            trace = go.Parcoords(
                line=dict(color='blue'),
                dimensions=dimensions,
                name=legend,
            )

        return trace
