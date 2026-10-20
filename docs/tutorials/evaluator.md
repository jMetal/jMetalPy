# Evaluate solutions

The lifecycle of metaheuristics often requires evaluating a list of solutions on every iteration.
In evolutionary algorithms, for example, this list of solutions is known as *population*.

In order to evaluate a population, NSGA-II (and in general, any generational algorithm in
jMetalPy) uses an evaluator object.

## Sequential

The default evaluator runs in a sequential fashion (i.e., one solution at a time):

```python
from jmetal.util.evaluator import SequentialEvaluator

algorithm = NSGAII(
   problem=problem,
   population_size=100,
   offspring_population_size=100,
   ...
   population_evaluator = SequentialEvaluator(),
 )
```

### API

::: jmetal.util.evaluator.SequentialEvaluator

## Cached evaluations

`EvaluatorWithCache` wraps any evaluator and avoids evaluating the same decision variables more
than once. It is useful when each evaluation calls an expensive simulator. Cache misses are passed
to the wrapped evaluator, so the same wrapper works with sequential and parallel evaluators:

```python
from jmetal.util.evaluator import EvaluatorWithCache, MultiprocessEvaluator

evaluator = EvaluatorWithCache(MultiprocessEvaluator(processes=8))

algorithm = NSGAII(
   problem=problem,
   population_size=100,
   offspring_population_size=100,
   ...
   population_evaluator=evaluator,
)
```

The wrapper restores both objective and constraint values for cached solutions. The algorithm's
evaluation counter still advances by the population size, including cache hits, so use a time-based
termination criterion when comparing the simulator work saved by caching.

### API

::: jmetal.util.evaluator.EvaluatorWithCache

## Parallel

Solutions can also be evaluated in parallel, using threads or processes:

```python
from jmetal.util.evaluator import MapEvaluator
from jmetal.util.evaluator import MultiprocessEvaluator
```

jMetalPy also includes evaluators based on Apache Spark and Dask, useful when a single solution
evaluation is itself expensive (e.g. simulation-based problems):

```python
from jmetal.util.evaluator import SparkEvaluator

algorithm = NSGAII(
   problem=problem,
   population_size=100,
   offspring_population_size=100,
   ...
   population_evaluator = SparkEvaluator(processes=8),
 )
```

Or by means of Dask:

```python
from jmetal.util.evaluator import DaskEvaluator

algorithm = NSGAII(
   problem=problem,
   population_size=100,
   offspring_population_size=100,
   ...
   population_evaluator = DaskEvaluator(),
 )
```

!!! warning
    `SparkEvaluator` and `DaskEvaluator` require pySpark and Dask, respectively (install via
    `pip install "jmetalpy[distributed]"`). Both currently run against a **local** Spark/Dask
    scheduler (`local[n]`) — they parallelize evaluation across the cores of one machine, not
    across a cluster, regardless of the `processes` argument.

### API

::: jmetal.util.evaluator.MapEvaluator

::: jmetal.util.evaluator.MultiprocessEvaluator

::: jmetal.util.evaluator.SparkEvaluator

::: jmetal.util.evaluator.DaskEvaluator
