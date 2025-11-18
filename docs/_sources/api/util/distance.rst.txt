Distance Metrics
================

Distance calculation utilities for multi-objective optimization problems.

Overview
--------

The distance module provides various distance metrics and calculation utilities optimized for different use cases in multi-objective optimization:

* **DistanceMetric**: Enumeration of available distance metrics
* **DistanceCalculator**: High-performance distance calculation utility
* **EuclideanDistance**: Standard Euclidean distance implementation
* **CosineDistance**: Cosine distance with reference point translation

Distance Metrics Enumeration
-----------------------------

.. currentmodule:: jmetal.util.distance

.. autoclass:: DistanceMetric
   :members:
   :undoc-members:
   :show-inheritance:

   Available distance metrics:

   * **L2_SQUARED**: Squared Euclidean distance (fastest, avoids sqrt computation)
   * **LINF**: L-infinity (Chebyshev) distance (efficient for high dimensions)
   * **TCHEBY_WEIGHTED**: Weighted Chebyshev distance (supports preferences)

Distance Calculator
-------------------

.. autoclass:: DistanceCalculator
   :members:
   :show-inheritance:

   High-performance static utility class for distance calculations. Supports multiple
   optimized metrics for different optimization scenarios.

   **Performance Characteristics:**

   * **L2_SQUARED**: ~15-20% faster than standard Euclidean distance
   * **LINF**: Efficient for high-dimensional objective spaces
   * **TCHEBY_WEIGHTED**: Flexible preference-based distance calculation

   **Usage Examples:**

   .. code-block:: python

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

Traditional Distance Classes
----------------------------

.. autoclass:: EuclideanDistance
   :members:
   :show-inheritance:

   Enhanced Euclidean distance implementation with comprehensive input validation
   and support for both Python lists and numpy arrays.

.. autoclass:: CosineDistance
   :members:
   :show-inheritance:

   Cosine distance implementation with reference point translation for specialized
   use cases in multi-objective optimization.

See Also
--------

* :doc:`../archive` - Archive implementations that use distance metrics
* :doc:`../../advanced-topics/distance-based-archive` - Tutorial on distance-based selection