Front visualization
========================

The :py:mod:`jmetal.lab.visualization` submodule contains several classes useful for plotting solutions. jMetalPy includes three types of visualization charts: static, interactive and streaming.

Static plots
------------------------

- It is possible to visualize the final front approximation by using the :code:`Plot` class:

  .. code-block:: python

     from jmetal.util.visualization import Plot

     front = algorithm.get_result()

     plot_front = Plot(plot_title='Pareto front approximation', axis_labels=['x', 'y'], reference_front=None)
     plot_front.plot(front, label='NSGAII-ZDT1', filename='NSGAII-ZDT1', format='png')

  .. note:: Static charts can be shown in the screen or stored in a file by setting the filename.

  For problems with two and three objectives, the figure produced is a scatter plot; for problems with more than three objectives,
  a parallel coordinates plot is used.
  Note that any arbitrary number of fronts can be plotted for comparison purposes:

  .. code-block:: python

     plot_front = Plot(plot_title='Title', axis_labels=['x', 'y'], reference_front=reference_front)
     plot_front.plot([front1, front2], label=['ProblemA', 'ProblemB'], filename='output', format='eps')

API
^^^

.. automodule:: jmetal.lab.visualization.plotting
   :members:
   :undoc-members:
   :show-inheritance:

Interactive plots
------------------------

- This kind of plots are interactive in such a way that every solution can be manipulated (e.g., actions such as zoom, selecting part of the graph, or clicking in a point to see its objective values are allowed).

  .. code-block:: python

     plot_front = InteractivePlot(plot_title='Pareto front approximation')
     plot_front.plot(front, label='NSGAII-ZDT1', filename='NSGAII-ZDT1-interactive')

API
^^^

.. automodule:: jmetal.lab.visualization.interactive
   :members:
   :undoc-members:
   :show-inheritance:

Streaming plots
------------------------

- The visualizer observer displays the front in real-time (note **it only works for problems with two and three objectives**) during the execution of multi-objective algorithms; this can be useful to observe the evolution of the Pareto front approximation:

  .. code-block:: python

     from jmetal.util.observer import VisualizerObserver

     algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front))

API
^^^

.. automodule:: jmetal.lab.visualization.streaming
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: S, pause

Chord plot
-------------------------------------------

.. automodule:: jmetal.lab.visualization.chord_plot
   :members:
   :undoc-members:
   :show-inheritance:

Posterior plot
-----------------------------------------

.. automodule:: jmetal.lab.visualization.posterior
   :members:
   :undoc-members:
   :show-inheritance:
