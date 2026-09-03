# Defining new problems

To include a problem in jMetalPy, it must implement the `Problem` interface from the
`jmetal.core.problem` module.

## Use case: Subset Sum

The goal is to find a subset S of W (list of non-negative integers) whose elements sum is closest
to (without exceeding) C. For example, for the input \(W=\{3, 34, 4, 12, 5, 2\}\) and \(C=9\), one
output could be \(S=\{4, 5\}\) (as it is a subset with sum 9).

In jMetalPy, this problem can be encoded as a binary problem with one objective (to be maximized)
and one bit per element of W, indicating whether that element is selected:

```python
import numpy as np

from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution


class SubsetSum(BinaryProblem):
   def __init__(self, C: int, W: list):
      super().__init__()
      self.C = C
      self.W = np.array(W, dtype=float)

      self.number_of_bits = len(self.W)
      self.obj_directions = [self.MAXIMIZE]
      self.obj_labels = ['Sum']

   def number_of_variables(self) -> int:
      return self.number_of_bits

   def number_of_objectives(self) -> int:
      return 1

   def number_of_constraints(self) -> int:
      return 0

   def evaluate(self, solution: BinarySolution) -> BinarySolution:
      pass

   def create_solution(self) -> BinarySolution:
      pass

   def name(self) -> str:
      return 'Subset Sum'
```

Now we have to define the abstract methods `evaluate` and `create_solution` from the
`jmetal.core.problem.Problem` class.

Note that each solution consists of one objective function to be maximized to be as close as
possible to \(C\):

\[
\max{\sum_{i \in S}{s_i}}
\]

Taking this into account, one solution could be created and evaluated as follows:

!!! note
    jMetalPy assumes minimization by default. Therefore, we will have to negate the solution
    objective.

```python
def evaluate(self, solution: BinarySolution) -> BinarySolution:
    selected_mask = solution.bits
    total_sum = np.sum(self.W[selected_mask])

    if total_sum > self.C:
        total_sum = self.C - (total_sum - self.C)
        if total_sum < 0.0:
            total_sum = 0.0

    solution.objectives[0] = -total_sum

    return solution

def create_solution(self) -> BinarySolution:
    solution = BinarySolution(
        number_of_variables=self.number_of_bits,
        number_of_objectives=self.number_of_objectives(),
    )
    solution.bits = np.random.choice([True, False], size=self.number_of_bits)

    return solution
```

`BinarySolution` stores its bits as a NumPy boolean array. Use the `bits` property to read or
assign the whole array at once (as above), or the list-based `variables` property for
element-by-element access.

## Use case: Multi-objective Subset Sum

The former problem can be formulated as a multi-objective binary problem whose objectives are as
follows:

1. Maximize the sum of subsets to be as close as possible to \(C\) and
2. Minimize the number of elements selected from \(W\).

This can be done by returning 2 from `number_of_objectives`, setting a second objective direction
and label, and computing both objectives in `evaluate`:

```diff
 class SubsetSum(BinaryProblem):
    def __init__(self, C: int, W: list):
       super().__init__()
       self.C = C
       self.W = np.array(W, dtype=float)

       self.number_of_bits = len(self.W)
-      self.obj_directions = [self.MAXIMIZE]
-      self.obj_labels = ['Sum']
+      self.obj_directions = [self.MAXIMIZE, self.MINIMIZE]
+      self.obj_labels = ['Sum', 'No. of Objects']

    def number_of_variables(self) -> int:
       return self.number_of_bits

    def number_of_objectives(self) -> int:
-      return 1
+      return 2

    def number_of_constraints(self) -> int:
       return 0

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
       selected_mask = solution.bits
       total_sum = np.sum(self.W[selected_mask])
+      number_of_objects = np.count_nonzero(selected_mask)

       if total_sum > self.C:
          total_sum = self.C - (total_sum - self.C)
          if total_sum < 0.0:
              total_sum = 0.0

       solution.objectives[0] = -total_sum
+      solution.objectives[1] = number_of_objects

       return solution
```

Both variants are available out of the box as
`jmetal.problem.singleobjective.unconstrained.SubsetSum` and
`jmetal.problem.multiobjective.unconstrained.SubsetSum`.

## API

::: jmetal.core.problem
