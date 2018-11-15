Visualization
========================

The :py:mod:`jmetal.util.graphic` module contains two classes useful for plotting solutions:

- :code:`FrontPlot` works for problems with any number of objectives:

  .. code-block:: python

     from jmetal.util import FrontPlot

     front = algorithm.get_result()

     pareto_front = FrontPlot(plot_title='Title', axis_labels=problem.obj_labels)
     pareto_front.plot(front, reference_front=problem.reference_front)
     pareto_front.to_html(filename='output')

  For problem with two and three objectives, the output is a scatter plot; for problem with more than three objectives, a parallel coordinates plot is used.
  In both two and three objectives the plot is interactive in such a way that every solution can be selected to print it.

  The plot can also be exported as an standalone *div* container for embedding the graph in an HTML file:

  .. code-block:: python

     pareto_front.export(filename='output', include_plotlyjs=False)

- :code:`ScatterStreaming` is intended to be used as an observer. The visualizer observer displays the front in real-time (although **it only works for problems with two and three objectives**):

  .. code-block:: python

     from jmetal.component import VisualizerObserver

     visualizer = VisualizerObserver()
     algorithm.observable.register(observer=visualizer)