Observers
========================

It is possible to attach any number of observers to a jMetalPy's algorithm to retrieve information from each iteration.
For example, a basic algorithm observer will print the number of evaluations, the objectives from the best individual in the population and the computing time:


.. code-block:: python

   observer = BasicAlgorithmObserver(frequency=1.0)
   algorithm.observable.register(observer=observer)

Write Front To File
------------------------------------

This observer will save each generation of individuals in files:

.. code-block:: python

   observer = WriteFrontToFileObserver(output_directory="FUN")
   algorithm.observable.register(observer=observer)

Visualization
------------------------------------

This observer will plot in real-time the Pareto frontier (with a reference front, if any) in 2D or 3D depending on the number of objectives.

.. code-block:: python

   observer = VisualizerObserver(frequency=1.0)
   algorithm.observable.register(observer=observer)