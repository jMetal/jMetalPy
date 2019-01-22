Front visualization
========================

The :py:mod:`jmetal.util.visualization` submodule contains several classes useful for plotting solutions. jMetalPy includes three types of visualization charts: static, interactive and streaming.

Static plots
------------------------

- It is possible to visualize the final front approximation by means of the :code:`Plot` class:

  .. code-block:: python

     from jmetal.util.visualization import Plot

     reference_front = ...
     front = algorithm.get_result()

     plot_front = Plot(plot_title='Pareto front approximation', axis_labels=['x', 'y'], reference_front=reference_front)
     plot_front.plot([front], label=['Problem'], filename='example', format='eps')

  .. note:: Static charts can be shown in the screen or stored in a file by setting the filename.

  For problems with two and three objectives, the output is a scatter plot; for problem with more than three objectives, a parallel coordinates plot is used.
  Note that any arbitrary number of fronts can be plotted for comparison purposes:

  .. code-block:: python

     plot_front = Plot(plot_title='Pareto front approximation', axis_labels=['x', 'y'], reference_front=reference_front)
     plot_front.plot([front1, front2], label=['ProblemA', 'ProblemB'], filename='output', format='eps')

Interactive plots
------------------------

- This kind of plots are interactive in such a way that every solution can be manipulated (e.g., actions such as zoom, selecting part of the graph, or clicking in a point to see its objective values are allowed).

  .. code-block:: python

     plot_front = InteractivePlot(plot_title='Pareto front approximation', axis_labels=['x', 'y'], reference_front=reference_front)
     plot_front.plot(front, filename='output')

  The plot can also be exported as an standalone *div* container for embedding the graph in an HTML file:

  .. code-block:: python

     plot_front.export_to_div(filename='div')

Streaming plots
------------------------

- This plot is intended to be used as an observer entity. The visualizer observer displays the front in real-time (although **it only works for problems with two and three objectives**) during the execution of the algorithms (and can also be included in a Jupyter notebook); this can be useful to observe the evolution of the current Pareto front approximation produced by the algorithm:

  .. code-block:: python

     from jmetal.util.observer import VisualizerObserver

     algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front))
