Distance-Based Archive Tutorial
===============================

This tutorial demonstrates how to use the ``DistanceBasedArchive`` class for maintaining diverse solution sets in multi-objective optimization problems.

Introduction
------------

The ``DistanceBasedArchive`` is an adaptive archive that automatically selects the best strategy based on the number of objectives:

* **2 objectives**: Uses crowding distance selection
* **>2 objectives**: Uses distance-based subset selection with normalization

This makes it particularly useful for problems where you don't know in advance how many objectives you'll be dealing with, or when working with both bi-objective and many-objective variants of the same problem.

Basic Usage
-----------

Creating and Using an Archive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jmetal.util.archive import DistanceBasedArchive
   from jmetal.core.solution import FloatSolution

   # Create archive with maximum size of 10
   archive = DistanceBasedArchive(maximum_size=10)

   # Create some example solutions
   solutions = []
   for i in range(20):
       solution = FloatSolution([], [], 2)  # 2 objectives
       solution.objectives = [i/20.0, 1.0 - i/20.0]  # Trade-off front
       solutions.append(solution)

   # Add solutions to archive
   for solution in solutions:
       was_added = archive.add(solution)
       print(f"Solution {solution.objectives} added: {was_added}")

   print(f"Final archive size: {archive.size()}")
   print("Selected solutions:")
   for i in range(archive.size()):
       sol = archive.get(i)
       print(f"  {sol.objectives}")

Custom Distance Metrics
~~~~~~~~~~~~~~~~~~~~~~~

The ``DistanceBasedArchive`` supports multiple distance metrics for enhanced performance and flexibility:

.. code-block:: python

   from jmetal.util.distance import DistanceMetric
   from jmetal.util.archive import DistanceBasedArchive
   import numpy as np

   # L2 squared distance (fastest, default)
   archive_l2 = DistanceBasedArchive(
       maximum_size=5,
       metric=DistanceMetric.L2_SQUARED
   )

   # Chebyshev distance (L-infinity)
   archive_linf = DistanceBasedArchive(
       maximum_size=5,
       metric=DistanceMetric.LINF
   )

   # Weighted Chebyshev distance (for preference-based selection)
   weights = np.array([0.5, 0.3, 0.2])  # Higher weight = more important
   archive_weighted = DistanceBasedArchive(
       maximum_size=5,
       metric=DistanceMetric.TCHEBY_WEIGHTED,
       weights=weights
   )

   # Deterministic results with fixed seed
   archive_reproducible = DistanceBasedArchive(
       maximum_size=5,
       random_seed=42
   )

Advanced Examples
-----------------

Two-Objective Problems
~~~~~~~~~~~~~~~~~~~~~~

For two-objective problems, the archive automatically uses crowding distance:

.. code-block:: python

   import numpy as np
   from jmetal.util.archive import DistanceBasedArchive
   from jmetal.core.solution import FloatSolution

   def create_zdt1_front(n_points=50):
       """Create solutions on ZDT1 Pareto front"""
       solutions = []
       for i in range(n_points):
           solution = FloatSolution([], [], 2)
           f1 = i / (n_points - 1)  # f1 in [0, 1]
           f2 = 1 - np.sqrt(f1)     # ZDT1 Pareto front
           solution.objectives = [f1, f2]
           solutions.append(solution)
       return solutions

   # Create archive and add Pareto front solutions
   archive = DistanceBasedArchive(maximum_size=10)
   solutions = create_zdt1_front(50)
   
   for solution in solutions:
       archive.add(solution)

   print(f"Selected {archive.size()} solutions from {len(solutions)} candidates")
   
   # Solutions will be selected to maximize crowding distance
   for i in range(archive.size()):
       sol = archive.get(i)
       crowding_dist = sol.attributes.get("crowding_distance", "N/A")
       print(f"Solution {i}: {sol.objectives}, crowding_distance: {crowding_dist}")

Many-Objective Problems
~~~~~~~~~~~~~~~~~~~~~~~

For problems with more than 2 objectives, distance-based selection is used:

.. code-block:: python

   import random
   from jmetal.util.archive import DistanceBasedArchive
   from jmetal.core.solution import FloatSolution

   def create_many_objective_solutions(n_solutions=100, n_objectives=5):
       """Create diverse solutions in many-objective space"""
       solutions = []
       random.seed(42)  # For reproducibility
       
       for i in range(n_solutions):
           solution = FloatSolution([], [], n_objectives)
           # Create solutions with different trade-offs
           objectives = []
           for j in range(n_objectives):
               # Some solutions excel in specific objectives
               if i % n_objectives == j:
                   objectives.append(random.uniform(0.0, 0.3))  # Good in this objective
               else:
                   objectives.append(random.uniform(0.4, 1.0))  # Worse in others
           solution.objectives = objectives
           solutions.append(solution)
       
       return solutions

   # Create archive for 5-objective problem
   archive = DistanceBasedArchive(maximum_size=10)
   solutions = create_many_objective_solutions(100, 5)

   for solution in solutions:
       archive.add(solution)

   print(f"Selected {archive.size()} solutions from {len(solutions)} candidates")
   print("Selected solutions (5 objectives):")
   for i in range(archive.size()):
       sol = archive.get(i)
       obj_str = [f"{obj:.3f}" for obj in sol.objectives]
       print(f"  Solution {i}: [{', '.join(obj_str)}]")

Standalone Subset Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Independent Usage of Selection Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also use the distance-based selection function independently with custom metrics:

.. code-block:: python

   from jmetal.util.archive import distance_based_subset_selection
   from jmetal.util.distance import DistanceMetric
   import numpy as np

   # Assuming you have a list of solutions
   solutions = create_many_objective_solutions(50, 4)

   # Select best 5 solutions using different metrics
   
   # L2 squared distance (fastest)
   selected_l2 = distance_based_subset_selection(
       solution_list=solutions,
       subset_size=5,
       metric=DistanceMetric.L2_SQUARED,
       random_seed=42  # For reproducible results
   )

   # Chebyshev distance (emphasizes worst-case differences)
   selected_linf = distance_based_subset_selection(
       solution_list=solutions,
       subset_size=5,
       metric=DistanceMetric.LINF,
       random_seed=42
   )

   # Weighted selection (prefer certain objectives)
   weights = np.array([0.4, 0.3, 0.2, 0.1])  # Prefer first objectives
   selected_weighted = distance_based_subset_selection(
       solution_list=solutions,
       subset_size=5,
       metric=DistanceMetric.TCHEBY_WEIGHTED,
       weights=weights,
       random_seed=42
   )

   print("L2 squared selection:")
   for i, sol in enumerate(selected_l2):
       print(f"  {i}: {[f'{obj:.3f}' for obj in sol.objectives]}")
   
   print("\\nChebyshev selection:")
   for i, sol in enumerate(selected_linf):
       print(f"  {i}: {[f'{obj:.3f}' for obj in sol.objectives]}")
   
   print("\\nWeighted selection:")
   for i, sol in enumerate(selected_weighted):
       print(f"  {i}: {[f'{obj:.3f}' for obj in sol.objectives]}")

Integration with Algorithms
----------------------------

The ``DistanceBasedArchive`` can be used with optimization algorithms that support archives:

.. code-block:: python

   from jmetal.algorithm.multiobjective.nsgaii import NSGAII
   from jmetal.util.archive import DistanceBasedArchive
   from jmetal.problem import ZDT1

   # Create algorithm with custom archive
   problem = ZDT1()
   archive = DistanceBasedArchive(maximum_size=100)

   # Note: This is conceptual - actual integration depends on algorithm design
   # Some algorithms may need modification to accept custom archives

Performance Tips
----------------

**Choosing Distance Metrics:**

* **L2_SQUARED**: Fastest option, good for general use (15-20% faster than standard Euclidean)
* **LINF**: Efficient for high-dimensional spaces, emphasizes worst-case differences
* **TCHEBY_WEIGHTED**: Use when objectives have different importance or scales

**For Better Performance:**

1. **Choose appropriate archive sizes**: Larger archives mean more comparisons
2. **Pre-filter dominated solutions**: Use ``NonDominatedSolutionsArchive`` first if needed
3. **Use L2_SQUARED metric**: Fastest for most cases due to avoided sqrt computation
4. **Set random seeds**: For reproducible results in deterministic environments

**Thread Safety:**

The ``DistanceBasedArchive`` is thread-safe for concurrent access:

.. code-block:: python

   import threading
   from jmetal.util.archive import DistanceBasedArchive

   # Safe for concurrent use
   archive = DistanceBasedArchive(maximum_size=100)

   def worker_thread(solutions_batch):
       for solution in solutions_batch:
           archive.add(solution)  # Thread-safe operation

**Memory Considerations:**

* Archives store references to solutions, not copies
* Large archives with many objectives can be memory-intensive
* Consider using archives as final result storage, not intermediate processing

Troubleshooting
---------------

**Common Issues:**

1. **Archive not filling up**: Check if solutions are being dominated
2. **Poor diversity**: Try different distance metrics or verify objective normalization
3. **Slow performance**: Use L2_SQUARED metric or smaller archive sizes
4. **Non-reproducible results**: Set ``random_seed`` parameter for deterministic behavior

**Debugging Example:**

.. code-block:: python

   from jmetal.util.solution import get_non_dominated_solutions
   from jmetal.util.archive import DistanceBasedArchive
   from jmetal.util.distance import DistanceMetric

   # Check if solutions are non-dominated
   non_dominated = get_non_dominated_solutions(your_solutions)
   print(f"Non-dominated solutions: {len(non_dominated)} out of {len(your_solutions)}")

   # Check archive behavior with debugging
   archive = DistanceBasedArchive(
       maximum_size=10, 
       metric=DistanceMetric.L2_SQUARED,
       random_seed=42  # For reproducible debugging
   )
   
   for i, solution in enumerate(your_solutions):
       was_added = archive.add(solution)
       print(f"Solution {i}: added={was_added}, archive_size={archive.size()}")
       
       if archive.size() > 0:
           latest_sol = archive.get(archive.size() - 1)
           print(f"  Latest solution objectives: {latest_sol.objectives}")

**Handling Constant Objectives:**

The archive automatically handles objectives with zero range (constant values):

.. code-block:: python

   # This works even if some objectives are constant
   solutions_with_constants = []
   for i in range(10):
       solution = FloatSolution([], [], 3)
       solution.objectives = [i * 0.1, 1.0, (9-i) * 0.1]  # Second objective constant
       solutions_with_constants.append(solution)
   
   archive = DistanceBasedArchive(maximum_size=5)
   for solution in solutions_with_constants:
       archive.add(solution)  # Handles constant objectives gracefully

See Also
--------

* :doc:`../api/util/archive` - Full API reference
* :doc:`../api/util/distance` - Distance measures  
* :doc:`../multiobjective.algorithms` - Algorithms that use archives
* :doc:`../tutorials/quality_indicators_cli` - Measuring solution quality