# Utilities

`jmetal.util` provides the archives, distance metrics, comparators, density estimators, and other
supporting utilities used throughout the framework. Evaluators, observers, and termination
criteria have their own tutorials with usage examples — see
[Evaluators](../tutorials/evaluator.md), [Observers](../tutorials/observer.md).

## Archives

Archives are data structures for storing and managing collections of solutions during the
optimization process. jMetalPy provides several implementations for different use cases:

- **Archive**: Base abstract class for all archives
- **BoundedArchive**: Archive with size limits
- **NonDominatedSolutionsArchive**: Maintains only non-dominated solutions
- **CrowdingDistanceArchive**: Uses crowding distance for diversity
- **DistanceBasedArchive**: Adaptive distance-based selection

### DistanceBasedArchive

`DistanceBasedArchive` provides adaptive selection strategies based on the number of objectives:

- **2 objectives**: Uses crowding distance selection for optimal diversity along Pareto fronts
- **>2 objectives**: Uses distance-based subset selection with normalization

**Key features:**

- Automatic strategy adaptation based on problem dimensionality
- Robust normalization handling constant objectives
- Support for custom distance measures
- Non-dominated solution filtering using Pareto dominance
- Memory-efficient in-place list modifications

**Algorithm for many-objective problems:**

1. Normalize objectives to [0,1] range using min-max normalization
2. Select random objective for initial sorting
3. Choose extreme solutions (best and worst in random objective)
4. Select remaining solutions using maximum minimum distance criterion

**Example usage:**

```python
from jmetal.util.archive import DistanceBasedArchive
from jmetal.util.distance import DistanceMetric

# Create archive with a custom distance metric
archive = DistanceBasedArchive(
    maximum_size=10,
    metric=DistanceMetric.L2_SQUARED,
)

# Add solutions - automatically adapts strategy
for solution in solutions:
    archive.add(solution)
```

::: jmetal.util.archive.DistanceBasedArchive

### Distance-based subset selection

Standalone function for distance-based subset selection that can be used independently of the
archive.

- **Selection strategy**: for 2 objectives, delegates to crowding distance selection; for more
  than 2, uses the distance-based algorithm with normalization.

```python
from jmetal.util.archive import distance_based_subset_selection

# Select 5 best solutions from a larger set
selected = distance_based_subset_selection(
    solution_list=all_solutions,
    subset_size=5,
)
```

::: jmetal.util.archive.distance_based_subset_selection

### Other archive classes

::: jmetal.util.archive.Archive

::: jmetal.util.archive.BoundedArchive

::: jmetal.util.archive.NonDominatedSolutionsArchive

::: jmetal.util.archive.CrowdingDistanceArchive

::: jmetal.util.archive.ArchiveWithReferencePoint

### Performance considerations

**Time complexity:**

- `DistanceBasedArchive.add()`: O(n²) for >2 objectives, O(n log n) for 2 objectives
- `distance_based_subset_selection()`: O(n²) worst case

**Space complexity:** O(n) for all archive implementations.

**Scalability:** Efficient for typical archive sizes (< 1000 solutions).

**Recommendations:**

- Use `CrowdingDistanceArchive` for 2-objective problems requiring only crowding distance
- Use `DistanceBasedArchive` for mixed or many-objective problems
- Use `NonDominatedSolutionsArchive` when size limits are not needed

## Distance metrics

The distance module provides various distance metrics and calculation utilities optimized for
different use cases in multi-objective optimization:

- **DistanceMetric**: Enumeration of available distance metrics — **L2_SQUARED** (squared
  Euclidean distance, fastest, avoids sqrt computation), **LINF** (L-infinity/Chebyshev distance,
  efficient for high dimensions), **TCHEBY_WEIGHTED** (weighted Chebyshev distance, supports
  preferences)
- **DistanceCalculator**: High-performance static utility class for distance calculations
- **EuclideanDistance**: Enhanced Euclidean distance with input validation for lists and arrays
- **CosineDistance**: Cosine distance with reference point translation

**Usage example:**

```python
import numpy as np
from jmetal.util.distance import DistanceCalculator, DistanceMetric

point1 = np.array([0.1, 0.5, 0.8])
point2 = np.array([0.3, 0.2, 0.9])

# L2 squared distance (fastest)
dist_l2 = DistanceCalculator.calculate_distance(
    point1, point2, DistanceMetric.L2_SQUARED
)

# Chebyshev distance
dist_linf = DistanceCalculator.calculate_distance(
    point1, point2, DistanceMetric.LINF
)

# Weighted Chebyshev distance
weights = np.array([0.5, 0.3, 0.2])
dist_weighted = DistanceCalculator.calculate_distance(
    point1, point2, DistanceMetric.TCHEBY_WEIGHTED, weights
)
```

::: jmetal.util.distance

## Comparators, rankings, and density estimators

::: jmetal.util.comparator

::: jmetal.util.ranking

::: jmetal.util.density_estimator

## Solutions, generators, and constraint handling

::: jmetal.util.solution

::: jmetal.util.generator

::: jmetal.util.constraint_handling

## Aggregation and neighborhoods (MOEA/D)

::: jmetal.util.aggregation_function

::: jmetal.util.neighborhood

## Normalization, points, and checks

::: jmetal.util.normalization

::: jmetal.util.point

::: jmetal.util.ckecking

## Observable

::: jmetal.util.observable

## Standalone plotting

A separate, CLI/benchmark-oriented plotting utility from `jmetal.lab.visualization` (see
[Front visualization](../tutorials/visualization.md)): reads a CSV of objective values and writes
a static PNG plus an optional interactive Plotly HTML file.

::: jmetal.util.plotting
