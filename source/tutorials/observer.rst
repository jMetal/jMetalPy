Extending the algorithms
========================

It is possible to attach any number of observers to any algorithm to get information from each iteration.
For example, a basic algorithm observer will print the number of evaluations, the objectives from the best individual in the population and the computing time:

.. code-block:: python

   basic = BasicAlgorithmObserver(frequency=1.0)
   algorithm.observable.register(observer=basic)

A progress bar observer will print a `smart progress meter <https://github.com/tqdm/tqdm>`_ that increases, on each iteration, a fixed value (`step`) until the maximum is reached.

.. code-block:: python

   max_evaluations = 25000

   algorithm = GeneticAlgorithm(...)

   progress_bar = ProgressBarObserver(max=max_evaluations)
   algorithm.observable.register(progress_bar)

   algorithm.run()

This will produce:

.. code-block:: console

   $ Progress:  50%|#####     | 12500/25000 [13:59<14:12, 14.66it/s]

A full list of all available observers can be found at the :py:mod:`jmetal.util.observer` module.

List of observers
-----------------------

.. automodule:: jmetal.util.observer
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: LOGGER

API
-----------------------

.. automodule:: jmetal.core.observer
   :members:
   :undoc-members:
   :show-inheritance: