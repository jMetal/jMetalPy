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

Custom Distance Measures
~~~~~~~~~~~~~~~~~~~~~~~~~

You can specify custom distance measures for the selection process:

.. code-block:: python

   from jmetal.util.distance import EuclideanDistance
   from jmetal.util.archive import DistanceBasedArchive

   # Create archive with explicit distance measure
   archive = DistanceBasedArchive(
       maximum_size=5,
       distance_measure=EuclideanDistance()
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

You can also use the distance-based selection function independently:

.. code-block:: python

   from jmetal.util.archive import distance_based_subset_selection
   from jmetal.util.distance import EuclideanDistance

   # Assuming you have a list of solutions
   solutions = create_many_objective_solutions(50, 4)

   # Select best 5 solutions using distance-based selection
   selected = distance_based_subset_selection(
       solution_list=solutions,
       subset_size=5,
       distance_measure=EuclideanDistance()
   )

   print(f"Selected {len(selected)} solutions:")
   for i, sol in enumerate(selected):
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

**For Better Performance:**

1. **Choose appropriate archive sizes**: Larger archives mean more comparisons
2. **Pre-filter dominated solutions**: Use ``NonDominatedSolutionsArchive`` first if needed
3. **Use efficient distance measures**: ``EuclideanDistance`` is usually fastest

**Memory Considerations:**

* Archives store references to solutions, not copies
* Large archives with many objectives can be memory-intensive
* Consider using archives as final result storage, not intermediate processing

Troubleshooting
---------------

**Common Issues:**

1. **Archive not filling up**: Check if solutions are being dominated
2. **Poor diversity**: Verify that objectives are properly normalized
3. **Slow performance**: Consider smaller archive sizes or simpler distance measures

**Debugging Example:**

.. code-block:: python

   # Check if solutions are non-dominated
   from jmetal.util.solution import get_non_dominated_solutions

   non_dominated = get_non_dominated_solutions(your_solutions)
   print(f"Non-dominated solutions: {len(non_dominated)} out of {len(your_solutions)}")

   # Check archive behavior
   archive = DistanceBasedArchive(maximum_size=10)
   for i, solution in enumerate(your_solutions):
       was_added = archive.add(solution)
       print(f"Solution {i}: added={was_added}, archive_size={archive.size()}")

See Also
--------

* :doc:`../api/util/archive` - Full API reference
* :doc:`../api/util/distance` - Distance measures
* :doc:`multiobjective-algorithms` - Algorithms that use archives
* :doc:`quality-indicators` - Measuring solution quality