Archives
========

Archives are data structures for storing and managing collections of solutions during the optimization process.

Overview
--------

jMetalPy provides several archive implementations for different use cases:

* **Archive**: Base abstract class for all archives
* **BoundedArchive**: Archive with size limits
* **NonDominatedSolutionsArchive**: Maintains only non-dominated solutions
* **CrowdingDistanceArchive**: Uses crowding distance for diversity
* **DistanceBasedArchive**: Adaptive distance-based selection

DistanceBasedArchive
--------------------

.. currentmodule:: jmetal.util.archive

.. autoclass:: DistanceBasedArchive
   :members:
   :show-inheritance:

   The ``DistanceBasedArchive`` class provides adaptive selection strategies based on the number of objectives:

   * **2 objectives**: Uses crowding distance selection for optimal diversity along Pareto fronts
   * **>2 objectives**: Uses distance-based subset selection with normalization

   **Key Features:**

   * Automatic strategy adaptation based on problem dimensionality
   * Robust normalization handling constant objectives
   * Support for custom distance measures
   * Non-dominated solution filtering using Pareto dominance
   * Memory-efficient in-place list modifications

   **Algorithm for Many-Objective Problems:**

   1. Normalize objectives to [0,1] range using min-max normalization
   2. Select random objective for initial sorting
   3. Choose extreme solutions (best and worst in random objective)
   4. Select remaining solutions using maximum minimum distance criterion

   **Example Usage:**

   .. code-block:: python

      from jmetal.util.archive import DistanceBasedArchive
      from jmetal.util.distance import EuclideanDistance
      
      # Create archive with custom distance measure
      archive = DistanceBasedArchive(
          maximum_size=10,
          distance_measure=EuclideanDistance()
      )
      
      # Add solutions - automatically adapts strategy
      for solution in solutions:
          archive.add(solution)

Distance-Based Subset Selection
-------------------------------

.. autofunction:: distance_based_subset_selection

   Standalone function for distance-based subset selection that can be used independently of the archive.

   **Parameters:**

   * ``solution_list``: List of solutions to select from
   * ``subset_size``: Number of solutions to select  
   * ``distance_measure``: Distance function (default: EuclideanDistance)

   **Selection Strategy:**

   * For 2 objectives: Delegates to crowding distance selection
   * For >2 objectives: Uses distance-based algorithm with normalization

   **Example:**

   .. code-block:: python

      from jmetal.util.archive import distance_based_subset_selection
      
      # Select 5 best solutions from a larger set
      selected = distance_based_subset_selection(
          solution_list=all_solutions,
          subset_size=5
      )

Other Archive Classes
---------------------

.. autoclass:: Archive
   :members:
   :show-inheritance:

.. autoclass:: BoundedArchive
   :members:
   :show-inheritance:

.. autoclass:: NonDominatedSolutionsArchive
   :members:
   :show-inheritance:

.. autoclass:: CrowdingDistanceArchive
   :members:
   :show-inheritance:

.. autoclass:: ArchiveWithReferencePoint
   :members:
   :show-inheritance:

Performance Considerations
--------------------------

**Time Complexity:**

* ``DistanceBasedArchive.add()``: O(n²) for >2 objectives, O(n log n) for 2 objectives
* ``distance_based_subset_selection()``: O(n²) worst case

**Space Complexity:** O(n) for all archive implementations

**Scalability:** Efficient for typical archive sizes (< 1000 solutions)

**Recommendations:**

* Use ``CrowdingDistanceArchive`` for 2-objective problems requiring only crowding distance
* Use ``DistanceBasedArchive`` for mixed or many-objective problems  
* Use ``NonDominatedSolutionsArchive`` when size limits are not needed

See Also
--------

* :doc:`../distance` - Distance measures and metrics
* :doc:`../comparator` - Solution comparison utilities
* :doc:`../normalization` - Objective normalization functions
* :doc:`../../algorithm/multiobjective` - Multi-objective algorithms using archives