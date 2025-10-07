"""
Example demonstrating the BestSolutionsArchive usage.

This script shows how to use the new BestSolutionsArchive class with both
2-objective and many-objective optimization problems.
"""

import random
from jmetal.core.solution import FloatSolution
from jmetal.util.archive import BestSolutionsArchive
from jmetal.util.distance import EuclideanDistance


def create_2d_pareto_front_solutions(num_solutions: int = 10):
    """Create a set of solutions forming a 2D Pareto front."""
    solutions = []
    for i in range(num_solutions):
        solution = FloatSolution([], [], 2)
        # Create trade-off between objectives (Pareto front)
        x = i / (num_solutions - 1)  # x in [0, 1]
        solution.objectives = [x, 1.0 - x]  # Trade-off front
        solutions.append(solution)
    return solutions


def create_many_objective_solutions(num_solutions: int = 20, num_objectives: int = 5):
    """Create a set of solutions for many-objective optimization."""
    solutions = []
    for i in range(num_solutions):
        solution = FloatSolution([], [], num_objectives)
        # Create diverse solutions in objective space
        objectives = []
        for j in range(num_objectives):
            # Each solution has different strengths in different objectives
            objectives.append(random.uniform(0.0, 1.0) + (i % num_objectives == j) * (-0.5))
        solution.objectives = objectives
        solutions.append(solution)
    return solutions


def demonstrate_2d_archive():
    """Demonstrate BestSolutionsArchive with 2-objective problems (uses crowding distance)."""
    print("=== 2-Objective Problem Demo ===")
    
    # Create archive with maximum size of 5
    archive = BestSolutionsArchive(maximum_size=5)
    
    # Generate 10 solutions on a Pareto front
    solutions = create_2d_pareto_front_solutions(10)
    
    print(f"Generated {len(solutions)} solutions")
    print("Solutions before filtering:")
    for i, sol in enumerate(solutions):
        print(f"  Solution {i}: {sol.objectives}")
    
    # Add solutions to archive
    for solution in solutions:
        archive.add(solution)
    
    print(f"\nArchive size after adding all solutions: {archive.size()}")
    print("Selected solutions (using crowding distance):")
    for i in range(archive.size()):
        sol = archive.get(i)
        crowding_dist = sol.attributes.get("crowding_distance", "N/A")
        print(f"  Solution {i}: {sol.objectives}, crowding_distance: {crowding_dist}")


def demonstrate_many_objective_archive():
    """Demonstrate BestSolutionsArchive with many-objective problems (uses distance-based selection)."""
    print("\n=== Many-Objective Problem Demo ===")
    
    # Create archive with maximum size of 5
    archive = BestSolutionsArchive(maximum_size=5, distance_measure=EuclideanDistance())
    
    # Generate 15 solutions in 4-objective space
    random.seed(42)  # For reproducible results
    solutions = create_many_objective_solutions(15, 4)
    
    print(f"Generated {len(solutions)} solutions in 4-objective space")
    print("First 5 solutions before filtering:")
    for i in range(min(5, len(solutions))):
        print(f"  Solution {i}: {[f'{obj:.3f}' for obj in solutions[i].objectives]}")
    
    # Add solutions to archive
    for solution in solutions:
        archive.add(solution)
    
    print(f"\nArchive size after adding all solutions: {archive.size()}")
    print("Selected solutions (using distance-based selection):")
    for i in range(archive.size()):
        sol = archive.get(i)
        obj_str = [f'{obj:.3f}' for obj in sol.objectives]
        print(f"  Solution {i}: {obj_str}")


def demonstrate_custom_distance():
    """Demonstrate using BestSolutionsArchive with custom distance measure."""
    print("\n=== Custom Distance Measure Demo ===")
    
    # Create archive with EuclideanDistance explicitly
    archive = BestSolutionsArchive(maximum_size=3, distance_measure=EuclideanDistance())
    
    # Create solutions that are best in each objective
    solutions = []
    for i in range(3):
        solution = FloatSolution([], [], 3)
        objectives = [1.0, 1.0, 1.0]
        objectives[i] = 0.0  # Best in objective i
        solution.objectives = objectives
        solutions.append(solution)
    
    # Add a few more diverse solutions
    for _ in range(3):
        solution = FloatSolution([], [], 3)
        solution.objectives = [random.uniform(0.2, 0.8) for _ in range(3)]
        solutions.append(solution)
    
    print(f"Generated {len(solutions)} solutions")
    print("All solutions:")
    for i, sol in enumerate(solutions):
        obj_str = [f'{obj:.3f}' for obj in sol.objectives]
        print(f"  Solution {i}: {obj_str}")
    
    # Add solutions to archive
    for solution in solutions:
        archive.add(solution)
    
    print(f"\nArchive size: {archive.size()}")
    print("Selected solutions:")
    for i in range(archive.size()):
        sol = archive.get(i)
        obj_str = [f'{obj:.3f}' for obj in sol.objectives]
        print(f"  Solution {i}: {obj_str}")


if __name__ == "__main__":
    print("BestSolutionsArchive Example")
    print("=" * 40)
    
    demonstrate_2d_archive()
    demonstrate_many_objective_archive()
    demonstrate_custom_distance()
    
    print("\nExample completed successfully!")