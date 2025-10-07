# BestSolutionsArchive Documentation

## Overview

The `BestSolutionsArchive` class is a sophisticated archive implementation that maintains the best solutions using distance-based subset selection. It extends the `BoundedArchive` functionality with an intelligent selection mechanism that adapts to the number of objectives in the optimization problem.

## Algorithm Description

The implementation follows the Java jMetal `BestSolutionsArchive` algorithm with the following strategy:

### For 2-Objective Problems
- Uses **crowding distance selection** 
- Leverages the existing `CrowdingDistanceArchive` implementation
- Selects solutions that maintain good diversity along the Pareto front

### For >2 Objective Problems  
- Uses **distance-based subset selection** with normalization
- Implements the following algorithm:
  1. Normalize all objectives to [0,1] range using min-max normalization
  2. Select a random objective for initial sorting
  3. Choose the first solution (best in random objective)
  4. Choose the last solution (worst in random objective) if subset_size > 1
  5. For remaining selections: choose solution with maximum minimum distance to already selected solutions

## Key Features

- **Adaptive Strategy**: Automatically chooses the best selection method based on problem dimensionality
- **Robust Normalization**: Handles constant objectives and edge cases gracefully
- **Distance Flexibility**: Supports custom distance measures (default: Euclidean)
- **Non-Dominated Filtering**: Maintains only non-dominated solutions using Pareto dominance
- **Memory Efficient**: Modifies solution lists in-place to maintain parent class references

## Usage Examples

### Basic Usage
```python
from jmetal.util.archive import BestSolutionsArchive
from jmetal.core.solution import FloatSolution

# Create archive with maximum size of 5
archive = BestSolutionsArchive(maximum_size=5)

# Add solutions
solution = FloatSolution([], [], 2)
solution.objectives = [1.0, 2.0]
archive.add(solution)

print(f"Archive size: {archive.size()}")
```

### Custom Distance Measure
```python
from jmetal.util.distance import EuclideanDistance

archive = BestSolutionsArchive(
    maximum_size=10, 
    distance_measure=EuclideanDistance()
)
```

### Access Solutions
```python
# Get specific solution
first_solution = archive.get(0)

# Get all solutions
all_solutions = archive.solution_list
```

## Implementation Details

### Class Hierarchy
```
Archive (Abstract)
  └── BoundedArchive
      └── BestSolutionsArchive
```

### Key Methods

#### `__init__(maximum_size, distance_measure=None, dominance_comparator=None)`
- **maximum_size**: Maximum number of solutions to maintain
- **distance_measure**: Distance function (default: EuclideanDistance)
- **dominance_comparator**: Comparator for dominance (default: DominanceComparator)

#### `add(solution) -> bool`
- Adds solution using non-dominated sorting and distance-based selection
- Returns True if solution was added or archive was modified
- Automatically applies subset selection when size exceeds maximum

#### `distance_based_subset_selection(solution_list, subset_size, distance_measure)`
- Standalone function for distance-based selection
- Can be used independently of the archive
- Supports both 2-objective and many-objective problems

## Performance Characteristics

- **Time Complexity**: O(n²) for distance-based selection, O(n log n) for crowding distance
- **Space Complexity**: O(n) for storing solutions
- **Scalability**: Efficient for typical archive sizes (< 1000 solutions)

## Testing

The implementation includes comprehensive tests covering:
- Basic archive functionality (18 test cases)
- Edge cases (empty lists, single solutions, constant objectives)
- Deterministic behavior with fixed random seeds
- Integration with existing jMetalPy components
- Custom distance measure support

Run tests with:
```bash
python -m pytest tests/util/test_best_solutions_archive.py -v
```

## Compatibility

- **Python Version**: 3.7+
- **Dependencies**: numpy, scipy (inherited from distance classes)
- **jMetalPy Integration**: Full compatibility with existing archives and algorithms
- **Migration**: Drop-in replacement for other bounded archives in most use cases

## See Also

- `CrowdingDistanceArchive`: For 2-objective problems specifically
- `NonDominatedSolutionsArchive`: For unlimited non-dominated archives  
- `EuclideanDistance`, `CosineDistance`: Distance measure options
- `examples/util/best_solutions_archive_example.py`: Complete usage example