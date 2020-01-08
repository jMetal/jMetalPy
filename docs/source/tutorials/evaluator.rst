Evaluate solutions
========================

The lifecycle of metaheuristics often requires to evaluate list of solutions on every iteration. In evolutionary algorithms, for example, this list of solutions is known as *population*.

In order to evaluate a population, NSGA-II (and in general, any generational algorithms in jMetalPy) uses an evaluator object.

Sequential
------------------------

The default evaluator runs in a sequential fashion (i.e., one solution at a time):

.. code-block:: python

   from jmetal.util.evaluator import SequentialEvaluator

   algorithm = NSGAII(
      problem=problem,
      population_size=100,
      offspring_population_size=100,
      ...
      population_evaluator = SequentialEvaluator(),
    )

API
^^^

.. autoclass:: jmetal.util.evaluator.SequentialEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

Parallel and distributed
------------------------

Solutions can also be evaluated in parallel, using threads or processes:

.. code-block:: python

   from jmetal.util.evaluator import MapEvaluator
   from jmetal.util.evaluator import MultiprocessEvaluator

jMetalPy includes an evaluator based on Apache Spark, so the solutions can be evaluated in a variety of parallel systems (multicores, clusters):

.. code-block:: python

   from jmetal.util.evaluator import SparkEvaluator

   algorithm = NSGAII(
      problem=problem,
      population_size=100,
      offspring_population_size=100,
      ...
      population_evaluator = SparkEvaluator(processes=8),
    )

Or by means of Dask:

.. code-block:: python

   from jmetal.util.evaluator import DaskEvaluator

   algorithm = NSGAII(
      problem=problem,
      population_size=100,
      offspring_population_size=100,
      ...
      population_evaluator = DaskEvaluator(),
    )

.. warning:: :code:`SparkEvaluator` and :code:`DaskEvaluator` requires pySpark and Dask, respectively.

API
^^^

.. autoclass:: jmetal.util.evaluator.MapEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: jmetal.util.evaluator.MultiprocessEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: jmetal.util.evaluator.SparkEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: jmetal.util.evaluator.DaskEvaluator
   :members:
   :undoc-members:
   :show-inheritance:
