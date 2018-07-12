Observers
========================

It is possible to attach any number of observers to a jMetalPy's algorithm to retrieve information from each iteration.
For example, a basic algorithm observer will print the number of evaluations, the objectives from the best individual in the population and the computing time:

.. code-block:: python

   basic = BasicAlgorithmObserver(frequency=1.0)
   algorithm.observable.register(observer=basic)

A full list of all available observer can be found at :py:mod:`jmetal.component.observer` module.