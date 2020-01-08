Extending algorithms
========================

In jMetalPy, algorithms maintain a list of dependents or *observers* which are notified automatically after each iteration (think about event listeners).
This is known as the **observer pattern** and can be used to extend the functionality of our algorithms by registering new observers.

For example, a basic observer will log the current number of evaluations, the objective(s) from the best individual in the population and the current computing time:

.. code-block:: python

   basic = BasicAlgorithmObserver(frequency=1.0)
   algorithm.observable.register(observer=basic)

A progress bar observer will print a `smart progress meter <https://github.com/tqdm/tqdm>`_ that increases, on each iteration, a fixed value (or `step`) until the maximum is reached:

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