import copy
import math
import random
from typing import List, Callable, Optional

import numpy as np

from jmetal.core.operator import Crossover
from jmetal.core.solution import (
    BinarySolution,
    CompositeSolution,
    FloatSolution,
    IntegerSolution,
    PermutationSolution,
    Solution,
)
from jmetal.util.ckecking import Check

# Type variables for generic type hints

"""
.. module:: crossover
   :platform: Unix, Windows
   :synopsis: Module implementing crossover operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class NullCrossover(Crossover[Solution, Solution]):
    """A no-operation crossover operator that simply returns copies of the parents.
    
    This operator is useful as a placeholder when no crossover is desired in an algorithm.
    It creates deep copies of the parent solutions without performing any genetic
    recombination. The number of parents and children is fixed at 2.
    
    Example:
        >>> from jmetal.operator import NullCrossover
        >>> from jmetal.core.solution import FloatSolution
        >>> 
        >>> # Create two test solutions
        >>> parent1 = FloatSolution([0], [1], 1)
        >>> parent2 = FloatSolution([0], [1], 1)
        >>> parent1.variables = [0.5]
        >>> parent2.variables = [1.5]
        >>> 
        >>> # Apply null crossover
        >>> crossover = NullCrossover()
        >>> offspring = crossover.execute([parent1, parent2])
        >>> 
        # Offspring are copies of parents
        >>> offspring[0].variables[0] == parent1.variables[0]
        True
        >>> offspring[1].variables[0] == parent2.variables[0]
        True
    """
    
    def __init__(self):
        """Initialize the null crossover operator with zero probability."""
        super(NullCrossover, self).__init__(probability=0.0)

    def execute(self, parents: List[Solution]) -> List[Solution]:
        """Execute the crossover operation.
        
        Args:
            parents: A list of exactly two parent solutions.
            
        Returns:
            A list containing deep copies of the parent solutions.
            
        Raises:
            Exception: If the number of parents is not exactly two.
        """
        if len(parents) != 2:
            raise Exception("The number of parents is not two: {}".format(len(parents)))
            
        # Create deep copies to avoid modifying the original parents
        return [copy.deepcopy(parent) for parent in parents]

    def get_number_of_parents(self) -> int:
        """Get the number of parent solutions required.
        
        Returns:
            int: Always returns 2, as this operator works with exactly two parents.
        """
        return 2

    def get_number_of_children(self) -> int:
        """Get the number of offspring solutions produced.
        
        Returns:
            int: Always returns 2, as this operator produces two offspring.
        """
        return 2

    def get_name(self) -> str:
        """Get the name of the operator.
        
        Returns:
            str: "Null crossover"
        """
        return "Null crossover"


class PMXCrossover(Crossover[PermutationSolution, PermutationSolution]):
    """Partially Mapped Crossover (PMX) for permutation problems.
    
    PMX is a specialized crossover operator designed for permutation-based representations,
    commonly used in problems like the Traveling Salesman Problem (TSP) and other ordering problems.
    
    The operator works by:
    1. Selecting two random cut points in the parent permutations
    2. Creating an offspring by copying the segment between the cut points from parent1
    3. Filling the remaining positions with the relative order of elements from parent2,
       while avoiding duplicates using a mapping relationship
    
    Args:
        probability: The probability of applying the crossover (0.0 to 1.0).
                    For each pair of parents, this probability determines
                    whether crossover is applied.
                    
    Example:
        >>> from jmetal.operator import PMXCrossover
        >>> from jmetal.core.solution import PermutationSolution
        >>> 
        >>> # Create two test solutions (permutation of [0,1,2,3,4])
        >>> parent1 = PermutationSolution(5, 1)
        >>> parent2 = PermutationSolution(5, 1)
        >>> parent1.variables = [0, 1, 2, 3, 4]
        >>> parent2.variables = [4, 3, 2, 1, 0]
        >>> 
        >>> # Apply PMX crossover (with probability 1.0 to ensure execution)
        >>> crossover = PMXCrossover(probability=1.0)
        >>> offspring = crossover.execute([parent1, parent2])
        >>> 
        # The offspring will be a mix of both parents while preserving the permutation property
        >>> all(x in offspring[0].variables for x in range(5))
        True
        
    Reference:
        Goldberg, D. E., & Lingle, R. (1985). Alleles, loci, and the traveling
        salesman problem. In Proceedings of the First International Conference on
        Genetic Algorithms and their Applications (pp. 154-159).
    """
    
    def __init__(self, probability: float):
        """Initialize the PMX crossover operator.
        
        Args:
            probability: Crossover probability between 0.0 and 1.0.
        """
        super(PMXCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[PermutationSolution]) -> List[PermutationSolution]:
        """Execute the PMX crossover operation.
        
        Args:
            parents: A list of exactly two parent solutions of type PermutationSolution.
            
        Returns:
            A list containing two offspring solutions.
            
        Raises:
            Exception: If the number of parents is not exactly two.
        """
        if len(parents) != 2:
            raise Exception("The number of parents is not two: {}".format(len(parents)))

        # Create new PermutationSolution instances with the correct parameters
        parent1, parent2 = parents
        offspring = [
            parent1.__class__(
                number_of_variables=parent1.number_of_variables,
                number_of_objectives=parent1.number_of_objectives,
                number_of_constraints=parent1.number_of_constraints
            ),
            parent2.__class__(
                number_of_variables=parent2.number_of_variables,
                number_of_objectives=parent2.number_of_objectives,
                number_of_constraints=parent2.number_of_constraints
            )
        ]
        # Copy the variables from parents to offspring
        offspring[0].variables = parent1.variables.copy()
        offspring[1].variables = parent2.variables.copy()
        
        # Only perform crossover with the specified probability
        if random.random() <= self.probability:
            permutation_length = parents[0].number_of_variables
            
            # Select two distinct random points for crossover
            point1, point2 = sorted(random.sample(range(permutation_length), 2))
            
            # Create directional mappings to resolve conflicts without cycles
            mapping_child1 = {}  # parent2 segment value -> parent1 segment value
            mapping_child2 = {}  # parent1 segment value -> parent2 segment value
            for i in range(point1, point2 + 1):
                value_parent1 = parents[0].variables[i]
                value_parent2 = parents[1].variables[i]
                mapping_child1[value_parent2] = value_parent1
                mapping_child2[value_parent1] = value_parent2
            
            # Apply PMX crossover
            for i in range(permutation_length):
                if i < point1 or i > point2:
                    # For positions outside the crossover points
                    val1 = parents[0].variables[i]
                    val2 = parents[1].variables[i]
                    
                    # Resolve mappings
                    while val1 in mapping_child1:
                        val1 = mapping_child1[val1]
                    while val2 in mapping_child2:
                        val2 = mapping_child2[val2]
                        
                    offspring[0].variables[i] = val1
                    offspring[1].variables[i] = val2
                else:
                    # Swap the segment between the points
                    offspring[0].variables[i] = parents[1].variables[i]
                    offspring[1].variables[i] = parents[0].variables[i]

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "Partially Matched crossover"


class CXCrossover(Crossover[PermutationSolution, PermutationSolution]):
    """Cycle Crossover (CX) for permutation-based solutions.
    
    Cycle Crossover is a specialized operator for permutation problems that preserves the absolute
    positions of elements from both parents. It works by identifying cycles between two parent
    permutations and creating offspring by alternating between the cycles of the parents.
    
    The algorithm works as follows:
    1. Start with the first parent and identify a cycle of positions where the elements
       alternate between the two parents
    2. For the first offspring, take elements from parent 1 at the cycle positions
       and from parent 2 at all other positions
    3. For the second offspring, do the opposite (parent 2 at cycle positions,
       parent 1 elsewhere)
    
    This operator is particularly useful for problems where the absolute position of elements
    is important, such as the Traveling Salesman Problem (TSP).
    
    Args:
        probability: Crossover probability (0.0 to 1.0). The probability that crossover
                    will be applied to a given pair of parents.
                    
    Example:
        >>> from jmetal.operator import CXCrossover
        >>> from jmetal.core.solution import PermutationSolution
        >>>
        >>> # Create two parent solutions (permutation of [0,1,2,3,4])
        >>> parent1 = PermutationSolution(5, 1)
        >>> parent2 = PermutationSolution(5, 1)
        >>> parent1.variables = [0, 1, 2, 3, 4]  # Identity permutation
        >>> parent2.variables = [4, 3, 2, 1, 0]  # Reverse permutation
        >>>
        >>> # Create CX crossover with probability 1.0
        >>> crossover = CXCrossover(probability=1.0)
        >>> offspring = crossover.execute([parent1, parent2])
        >>>
        # The offspring will preserve absolute positions from both parents
        >>> all(x in offspring[0].variables for x in range(5))  # Still a valid permutation
        True
        
    Reference:
        Oliver, I. M., Smith, D. J., & Holland, J. R. (1987). A study of permutation
        crossover operators on the traveling salesman problem. In Proceedings of the
        Second International Conference on Genetic Algorithms on Genetic algorithms
        and their application (pp. 224-230).
    """
    
    def __init__(self, probability: float):
        """Initialize the Cycle Crossover operator.
        
        Args:
            probability: Crossover probability between 0.0 and 1.0.
        """
        super(CXCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[PermutationSolution]) -> List[PermutationSolution]:
        """Execute the Cycle Crossover operation.
        
        Args:
            parents: A list of exactly two parent solutions of type PermutationSolution.
                    Both parents must have the same length and contain the same elements.
            
        Returns:
            A list containing two offspring solutions.
            
        Raises:
            Exception: If the number of parents is not exactly two.
        """
        if len(parents) != 2:
            raise Exception("The number of parents is not two: {}".format(len(parents)))

        # Create copies of parents (swapped) to serve as offspring
        offspring = [copy.deepcopy(parents[1]), copy.deepcopy(parents[0])]
        
        # Only perform crossover with the specified probability
        if random.random() <= self.probability:
            # Start with a random position
            start_idx = random.randint(0, len(parents[0].variables) - 1)
            curr_idx = start_idx
            cycle = []

            # Find the cycle of positions
            while True:
                cycle.append(curr_idx)
                # Find where parent1's element is in parent2
                curr_idx = parents[0].variables.index(parents[1].variables[curr_idx])
                if curr_idx == start_idx:  # Completed a full cycle
                    break

            # Apply the cycle to create offspring
            for j in range(len(parents[0].variables)):
                if j in cycle:
                    # Take values from parent1 for cycle positions in offspring1
                    # and from parent2 for cycle positions in offspring2
                    offspring[0].variables[j] = parents[0].variables[j]
                    offspring[1].variables[j] = parents[1].variables[j]

        return offspring

    def get_number_of_parents(self) -> int:
        """Get the number of parent solutions required.
        
        Returns:
            int: Always returns 2, as this operator works with exactly two parents.
        """
        return 2

    def get_number_of_children(self) -> int:
        """Get the number of offspring solutions produced.
        
        Returns:
            int: Always returns 2, as this operator produces two offspring.
        """
        return 2

    def get_name(self) -> str:
        """Get the name of the operator.
        
        Returns:
            str: "Cycle crossover"
        """
        return "Cycle crossover"


class SBXCrossover(Crossover[FloatSolution, FloatSolution]):
    """Simulated Binary Crossover (SBX) for real-valued solutions.
    
    SBX is a popular crossover operator for real-coded genetic algorithms that simulates the behavior of the single-point
    crossover operator in binary-coded GAs. It creates offspring solutions based on a probability distribution centered
    around the parent solutions, with the spread of the distribution controlled by the distribution index.
    
    The operator works by:
    1. For each variable, compute a spread factor β based on a random number and the distribution index
    2. Use β to compute new variable values that are spread around the parent values
    3. The distribution index controls whether offspring are likely to be near the parents (high values)
       or more spread out (low values)
    
    Args:
        probability: Crossover probability (0.0 to 1.0). The probability that crossover will be applied
            to a given pair of parents.
        distribution_index: Distribution index (must be ≥ 0). Controls the shape of the probability distribution:
            - High values (>20): Offspring are very close to parents
            - Medium values (~10-20): Balanced exploration/exploitation
            - Low values (<5): High exploration, offspring can be far from parents
            Typical values range from 5 to 30, with 20 being a common default.
    
    Raises:
        ValueError: If distribution_index is negative
        
    Example:
        >>> from jmetal.operator import SBXCrossover
        >>> from jmetal.core.solution import FloatSolution
        >>>
        >>> # Create two parent solutions
        >>> parent1 = FloatSolution([0, 0], [1, 1], 1)
        >>> parent2 = FloatSolution([0, 0], [1, 1], 1)
        >>> parent1.variables = [0.2, 0.8]
        >>> parent2.variables = [0.8, 0.2]
        >>>
        >>> # Create SBX crossover with probability 0.9 and distribution index 20
        >>> crossover = SBXCrossover(probability=0.9, distribution_index=20.0)
        >>>
        >>> # Generate offspring
        >>> offspring = crossover.execute([parent1, parent2])
        >>> # Offspring will be similar to parents due to high distribution index
        >>> all(0.1 < x < 0.9 for x in offspring[0].variables + offspring[1].variables)
        True
    
    References:
        Deb, K., & Agrawal, R. B. (1995). Simulated binary crossover for continuous search space.
        Complex Systems, 9(2), 115-148.
        
        Deb, K., & Deb, K. (2014). Multi-objective optimization. In Search methodologies (pp. 403-449).
        Springer, Boston, MA.
    """
    __EPS = 1.0e-14  # Small constant to prevent division by zero

    def __init__(self, probability: float, distribution_index: float = 20.0):
        super(SBXCrossover, self).__init__(probability=probability)
        self.distribution_index = distribution_index
        if distribution_index < 0:
            raise ValueError("The distribution index cannot be negative")

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        Check.that(issubclass(type(parents[0]), FloatSolution), "Solution type invalid: " + str(type(parents[0])))
        Check.that(issubclass(type(parents[1]), FloatSolution), "Solution type invalid")
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))

        offspring = copy.deepcopy(parents)
        rand = random.random()

        if rand <= self.probability:
            for i in range(len(parents[0].variables)):
                value_x1, value_x2 = parents[0].variables[i], parents[1].variables[i]

                if random.random() <= 0.5:
                    if abs(value_x1 - value_x2) > self.__EPS:
                        if value_x1 < value_x2:
                            y1, y2 = value_x1, value_x2
                        else:
                            y1, y2 = value_x2, value_x1

                        # Calculate beta and handle potential complex numbers
                        try:
                            # First offspring (based on first parent's bounds)
                            lb1, ub1 = parents[0].lower_bound[i], parents[0].upper_bound[i]
                            beta1 = 1.0 + (2.0 * (y1 - lb1) / (y2 - y1))
                            alpha1 = 2.0 - pow(beta1, -(self.distribution_index + 1.0))
                            
                            rand_val = random.random()
                            if rand_val <= (1.0 / alpha1):
                                betaq1 = pow(rand_val * alpha1, (1.0 / (self.distribution_index + 1.0)))
                            else:
                                betaq1 = pow(1.0 / (2.0 - rand_val * alpha1), 1.0 / (self.distribution_index + 1.0))
                            
                            c1 = 0.5 * (y1 + y2 - betaq1 * (y2 - y1))
                            
                            # Ensure c1 is a real number and within bounds
                            if isinstance(c1, complex):
                                c1 = y1 if c1.real < lb1 else (y2 if c1.real > ub1 else c1.real)
                            
                            # Second offspring (based on second parent's bounds)
                            lb2, ub2 = parents[1].lower_bound[i], parents[1].upper_bound[i]
                            beta2 = 1.0 + (2.0 * (ub2 - y2) / (y2 - y1))
                            alpha2 = 2.0 - pow(beta2, -(self.distribution_index + 1.0))
                            
                            if rand_val <= (1.0 / alpha2):
                                betaq2 = pow((rand_val * alpha2), (1.0 / (self.distribution_index + 1.0)))
                            else:
                                betaq2 = pow(1.0 / (2.0 - rand_val * alpha2), 1.0 / (self.distribution_index + 1.0))
                            
                            c2 = 0.5 * (y1 + y2 + betaq2 * (y2 - y1))
                            
                            # Ensure c2 is a real number and within bounds
                            if isinstance(c2, complex):
                                c2 = y1 if c2.real < lb2 else (y2 if c2.real > ub2 else c2.real)
                                
                        except (ValueError, ZeroDivisionError):
                            # Fallback to parent values if any numerical issues occur
                            c1, c2 = y1, y2

                        # Apply bounds checking using the correct bounds for each offspring
                        if c1 < lb1:
                            c1 = lb1
                        if c2 < lb2:
                            c2 = lb2
                        if c1 > ub1:
                            c1 = ub1
                        if c2 > ub2:
                            c2 = ub2

                        if random.random() <= 0.5:
                            offspring[0]._variables[i] = c2
                            offspring[1]._variables[i] = c1
                        else:
                            offspring[0]._variables[i] = c1
                            offspring[1]._variables[i] = c2
                    else:
                        offspring[0]._variables[i] = value_x1
                        offspring[1]._variables[i] = value_x2
                else:
                    offspring[0]._variables[i] = value_x1
                    offspring[1]._variables[i] = value_x2
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "SBX crossover"


class IntegerSBXCrossover(Crossover[IntegerSolution, IntegerSolution]):
    __EPS = 1.0e-14

    def __init__(self, probability: float, distribution_index: float = 20.0):
        super(IntegerSBXCrossover, self).__init__(probability=probability)
        self.distribution_index = distribution_index

    def execute(self, parents: List[IntegerSolution]) -> List[IntegerSolution]:
        Check.that(issubclass(type(parents[0]), IntegerSolution), "Solution type invalid")
        Check.that(issubclass(type(parents[1]), IntegerSolution), "Solution type invalid")
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))

        offspring = copy.deepcopy(parents)
        rand = random.random()

        if rand <= self.probability:
            for i in range(len(parents[0].variables)):
                value_x1, value_x2 = parents[0].variables[i], parents[1].variables[i]

                if random.random() <= 0.5:
                    if abs(value_x1 - value_x2) > self.__EPS:
                        if value_x1 < value_x2:
                            y1, y2 = value_x1, value_x2
                        else:
                            y1, y2 = value_x2, value_x1

                        # Calculate beta and handle potential complex numbers
                        try:
                            # First offspring (based on first parent's bounds)
                            lb1, ub1 = parents[0].lower_bound[i], parents[0].upper_bound[i]
                            beta1 = 1.0 + (2.0 * (y1 - lb1) / (y2 - y1))
                            alpha1 = 2.0 - pow(beta1, -(self.distribution_index + 1.0))
                            
                            rand_val = random.random()
                            if rand_val <= (1.0 / alpha1):
                                betaq1 = pow(rand_val * alpha1, (1.0 / (self.distribution_index + 1.0)))
                            else:
                                betaq1 = pow(1.0 / (2.0 - rand_val * alpha1), 1.0 / (self.distribution_index + 1.0))
                            
                            c1 = 0.5 * (y1 + y2 - betaq1 * (y2 - y1))
                            
                            # Ensure c1 is a real number and within bounds
                            if isinstance(c1, complex):
                                c1 = y1 if c1.real < lb1 else (y2 if c1.real > ub1 else c1.real)
                            
                            # Second offspring (based on second parent's bounds)
                            lb2, ub2 = parents[1].lower_bound[i], parents[1].upper_bound[i]
                            beta2 = 1.0 + (2.0 * (ub2 - y2) / (y2 - y1))
                            alpha2 = 2.0 - pow(beta2, -(self.distribution_index + 1.0))
                            
                            if rand_val <= (1.0 / alpha2):
                                betaq2 = pow((rand_val * alpha2), (1.0 / (self.distribution_index + 1.0)))
                            else:
                                betaq2 = pow(1.0 / (2.0 - rand_val * alpha2), 1.0 / (self.distribution_index + 1.0))
                            
                            c2 = 0.5 * (y1 + y2 + betaq2 * (y2 - y1))
                            
                            # Ensure c2 is a real number and within bounds
                            if isinstance(c2, complex):
                                c2 = y1 if c2.real < lb2 else (y2 if c2.real > ub2 else c2.real)
                                
                        except (ValueError, ZeroDivisionError):
                            # Fallback to parent values if any numerical issues occur
                            c1, c2 = y1, y2

                        # Apply bounds checking using the correct bounds for each offspring
                        if c1 < lb1:
                            c1 = lb1
                        if c2 < lb2:
                            c2 = lb2
                        if c1 > ub1:
                            c1 = ub1
                        if c2 > ub2:
                            c2 = ub2

                        if random.random() <= 0.5:
                            offspring[0]._variables[i] = int(c2)
                            offspring[1]._variables[i] = int(c1)
                        else:
                            offspring[0]._variables[i] = int(c1)
                            offspring[1]._variables[i] = int(c2)
                    else:
                        offspring[0]._variables[i] = value_x1
                        offspring[1]._variables[i] = value_x2
                else:
                    offspring[0]._variables[i] = value_x1
                    offspring[1]._variables[i] = value_x2
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "Integer SBX crossover"


class SPXCrossover(Crossover[BinarySolution, BinarySolution]):
    """
    A high-performance single-point crossover operator for BinarySolution.
    
    This implementation uses NumPy's vectorized operations for better performance
    when working with BinarySolution solutions. It performs a single-point
    crossover between two parent solutions to produce two offspring.
    
    The crossover point is selected uniformly at random from all possible bit
    positions in the solution. The bits after the crossover point are swapped
    between the two parents to create the offspring.
    
    Args:
        probability: The probability of applying the crossover (must be between 0.0 and 1.0)
        
    Raises:
        ValueError: If the probability is not in the range [0.0, 1.0]
    """
    
    def __init__(self, probability: float):
        if not (0.0 <= probability <= 1.0):
            raise ValueError(f"Probability must be between 0.0 and 1.0, but was {probability}")
        super(SPXCrossover, self).__init__(probability=probability)
        self._rng = np.random.default_rng()

    def execute(self, parents: List[BinarySolution]) -> List[BinarySolution]:
        """
        Execute the single-point crossover operation.
        
        Args:
            parents: A list of exactly two parent solutions of type BinarySolution.
                    Both parents must have the same number of bits.
            
        Returns:
            List[BinarySolution]: A list containing two offspring solutions.
            
        Note:
            This method assumes that both parents are valid BinarySolution instances
            with properly initialized bits attributes.
        """
        if len(parents) != 2:
            raise ValueError("SPXCrossover requires exactly two parents")
            
        # Create deep copies of the parents to avoid modifying the originals
        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        
        # Check if crossover should be performed based on probability
        if self._rng.random() > self.probability:
            return offspring
            
        # Get the bits from both parents
        bits1 = offspring[0].bits
        bits2 = offspring[1].bits
        
        # Ensure both parents have the same number of bits
        if len(bits1) != len(bits2):
            raise ValueError("Parents must have the same number of bits")
            
        num_bits = len(bits1)
        if num_bits > 1:
            # Select a random crossover point (1 to num_bits-1 to ensure crossover happens)
            crossover_point = self._rng.integers(1, num_bits)
            
            # Create new bit arrays for the offspring
            new_bits1 = np.concatenate([bits1[:crossover_point], bits2[crossover_point:]])
            new_bits2 = np.concatenate([bits2[:crossover_point], bits1[crossover_point:]])
            
            # Update the bits in the offspring
            offspring[0].bits = new_bits1
            offspring[1].bits = new_bits2
        
        return offspring

    def get_number_of_parents(self) -> int:
        """Return the number of parent solutions required by the operator."""
        return 2

    def get_number_of_children(self) -> int:
        """Return the number of offspring produced by the operator."""
        return 2

    def get_name(self) -> str:
        """Return the name of the operator."""
        return "Single point crossover"

class BLXAlphaCrossover(Crossover[FloatSolution, FloatSolution]):
    """BLX-α (Blend Crossover) for real-valued solutions.
    
    The BLX-α crossover creates offspring within a range that is extended by a factor of α (alpha)
    beyond the range defined by the parent values. This allows for exploration beyond the region
    defined by the parents while maintaining a balance between exploration and exploitation.
    
    The crossover works by:
    1. For each variable, determine the min and max values from the parents
    2. Calculate the range between parents
    3. Expand the range by α * range in both directions
    4. Sample new values uniformly from this expanded range
    5. Apply bounds repair if values fall outside the variable bounds
    
    Args:
        probability: Crossover probability (0.0 to 1.0)
        alpha: Expansion factor (must be ≥ 0). Controls the exploration range:
            - alpha = 0: Offspring will be in the range defined by parents (no exploration)
            - alpha > 0: Offspring can be outside parent range (increased exploration)
            - Typical values: 0.1 to 0.5
        repair_operator: Optional function to repair out-of-bounds values.
            If None, values are clamped to the variable bounds using min/max.
            Signature: repair_operator(value: float, lower_bound: float, upper_bound: float) -> float
    
    Raises:
        ValueError: If probability is not in [0,1] or alpha is negative.
        
    Reference:
        Eshelman, L. J., & Schaffer, J. D. (1993). Real-coded genetic algorithms and 
        interval-schemata. Foundations of genetic algorithms, 2, 187-202.
    """

    def __init__(
        self,
        probability: float = 0.9,
        alpha: float = 0.5,
        repair_operator: Optional[Callable[[float, float, float], float]] = None,
    ):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")

        super().__init__(probability=probability)
        self.alpha = alpha
        self.repair_operator = repair_operator if repair_operator else self._default_repair

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        Check.that(len(parents) == 2, "BLXAlphaCrossover requires exactly two parents")
        return self.doCrossover(self.probability, parents[0], parents[1])

    def doCrossover(
        self, probability: float, parent1: FloatSolution, parent2: FloatSolution
    ) -> List[FloatSolution]:
        """Perform the crossover operation.

        Args:
            probability: Crossover probability
            parent1: First parent solution
            parent2: Second parent solution

        Returns:
            A list containing two offspring solutions
        """
        offspring1 = parent1.__class__(
            parent1.lower_bound, 
            parent1.upper_bound, 
            len(parent1.objectives),
            len(parent1.constraints) if hasattr(parent1, 'constraints') else 0
        )
        offspring2 = parent2.__class__(
            parent2.lower_bound, 
            parent2.upper_bound, 
            len(parent2.objectives),
            len(parent2.constraints) if hasattr(parent2, 'constraints') else 0
        )

        if random.random() > probability:
            offspring1.variables = parent1.variables.copy()
            offspring2.variables = parent2.variables.copy()
            return [offspring1, offspring2]

        for i in range(len(parent1.variables)):
            x1, x2 = parent1.variables[i], parent2.variables[i]
            lower_bound = parent1.lower_bound[i]
            upper_bound = parent1.upper_bound[i]

            # Calculate the range between parents
            min_val = min(x1, x2)
            max_val = max(x1, x2)
            range_val = max_val - min_val

            # Expand the range by alpha
            min_range = min_val - range_val * self.alpha
            max_range = max_val + range_val * self.alpha

            # Generate offspring values within the expanded range
            y1 = random.uniform(min_range, max_range)
            y2 = random.uniform(min_range, max_range)

            # Repair out-of-bounds values
            y1 = self.repair_operator(y1, lower_bound, upper_bound)
            y2 = self.repair_operator(y2, lower_bound, upper_bound)

            offspring1._variables[i] = y1
            offspring2._variables[i] = y2

        return [offspring1, offspring2]

    def _default_repair(self, value: float, lower_bound: float, upper_bound: float) -> float:
        """Default repair method that clamps values to bounds."""
        return max(lower_bound, min(upper_bound, value))

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return f"BLX-alpha crossover"


class BLXAlphaBetaCrossover(Crossover[FloatSolution, FloatSolution]):
    """BLX-αβ (Blend Crossover with separate alpha and beta) for real-valued solutions.
    
    An extension of BLX-α crossover that uses two different expansion factors (α and β)
    for the lower and upper bounds respectively. This allows for asymmetric exploration
    around the parent solutions.
    
    The crossover works by:
    1. For each variable, determine the min and max values from the parents
    2. Calculate the range between parents (d = max - min)
    3. Expand the range by α*d below the min and β*d above the max
    4. Sample new values uniformly from this expanded range
    5. Apply bounds repair if values fall outside the variable bounds
    
    Args:
        probability: Crossover probability (0.0 to 1.0)
        alpha: Lower expansion factor (must be ≥ 0). Controls exploration below parents:
            - alpha = 0: No exploration below the smaller parent value
            - alpha > 0: Expands range below smaller parent by alpha*d
            - Typical values: 0.1 to 0.5
        beta: Upper expansion factor (must be ≥ 0). Controls exploration above parents:
            - beta = 0: No exploration above the larger parent value
            - beta > 0: Expands range above larger parent by beta*d
            - Typical values: 0.1 to 0.5
        repair_operator: Optional function to repair out-of-bounds values.
            If None, values are clamped to the variable bounds using min/max.
            Signature: repair_operator(value: float, lower_bound: float, upper_bound: float) -> float
    
    Raises:
        ValueError: If probability is not in [0,1] or alpha/beta are negative.
        
    Reference:
        Eshelman, L. J., & Schaffer, J. D. (1993). Real-coded genetic algorithms and 
        interval-schemata. Foundations of genetic algorithms, 2, 187-202.
    """

    def __init__(
        self,
        probability: float = 0.9,
        alpha: float = 0.5,
        beta: float = 0.5,
        repair_operator: Optional[Callable[[float, float, float], float]] = None,
    ):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if beta < 0:
            raise ValueError("beta must be non-negative")

        super().__init__(probability=probability)
        self.alpha = alpha
        self.beta = beta
        self.repair_operator = repair_operator if repair_operator else self._default_repair

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        Check.that(len(parents) == 2, "BLXAlphaBetaCrossover requires exactly two parents")
        return self.doCrossover(self.probability, parents[0], parents[1])

    def doCrossover(
        self, probability: float, parent1: FloatSolution, parent2: FloatSolution
    ) -> List[FloatSolution]:
        """Perform the crossover operation.

        Args:
            probability: Crossover probability
            parent1: First parent solution
            parent2: Second parent solution

        Returns:
            A list containing two offspring solutions
        """
        offspring1 = parent1.__class__(
            parent1.lower_bound, 
            parent1.upper_bound, 
            len(parent1.objectives),
            len(parent1.constraints) if hasattr(parent1, 'constraints') else 0
        )
        offspring2 = parent2.__class__(
            parent2.lower_bound, 
            parent2.upper_bound, 
            len(parent2.objectives),
            len(parent2.constraints) if hasattr(parent2, 'constraints') else 0
        )

        if random.random() > probability:
            offspring1.variables = parent1.variables.copy()
            offspring2.variables = parent2.variables.copy()
            return [offspring1, offspring2]

        for i in range(len(parent1.variables)):
            x1, x2 = parent1.variables[i], parent2.variables[i]
            lower_bound = parent1.lower_bound[i]
            upper_bound = parent1.upper_bound[i]

            # Ensure x1 <= x2
            if x1 > x2:
                x1, x2 = x2, x1

            # Calculate the range and expanded bounds
            d = x2 - x1
            c_min = x1 - self.alpha * d
            c_max = x2 + self.beta * d

            # Generate offspring values within the expanded range
            y1 = random.uniform(c_min, c_max)
            y2 = random.uniform(c_min, c_max)

            # Repair out-of-bounds values
            y1 = self.repair_operator(y1, lower_bound, upper_bound)
            y2 = self.repair_operator(y2, lower_bound, upper_bound)

            offspring1._variables[i] = y1
            offspring2._variables[i] = y2

        return [offspring1, offspring2]

    def _default_repair(self, value: float, lower_bound: float, upper_bound: float) -> float:
        """Default repair method that clamps values to bounds."""
        return max(lower_bound, min(upper_bound, value))

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return f"BLX-alpha-beta crossover (α={self.alpha}, β={self.beta})"


class ArithmeticCrossover(Crossover[FloatSolution, FloatSolution]):
    """Arithmetic Crossover for real-valued solutions.
    
    This operator performs an arithmetic combination of two parent solutions to produce
    two offspring. For each variable, a random weight (alpha) is used to compute a weighted
    average of the parent values.
    
    The crossover works by:
    1. For each variable, generate a random weight alpha in [0, 1]
    2. Calculate new values as:
       - child1 = alpha * parent1 + (1 - alpha) * parent2
       - child2 = (1 - alpha) * parent1 + alpha * parent2
    3. Apply bounds repair if values fall outside the variable bounds
    
    Args:
        probability: Crossover probability (0.0 to 1.0)
        repair_operator: Optional function to repair out-of-bounds values.
            If None, values are clamped to the variable bounds using min/max.
            Signature: repair_operator(value: float, lower_bound: float, upper_bound: float) -> float
    
    Raises:
        ValueError: If probability is not in [0,1]
        
    Reference:
        Michalewicz, Z. (1996). Genetic Algorithms + Data Structures = Evolution Programs.
        Springer-Verlag, Berlin.
    """

    def __init__(
        self,
        probability: float = 0.9,
        repair_operator: Optional[Callable[[float, float, float], float]] = None,
    ):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")

        super().__init__(probability=probability)
        self.repair_operator = repair_operator if repair_operator else self._default_repair

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        Check.that(len(parents) == 2, "Arithmetic Crossover requires exactly two parents")
        return self.doCrossover(self.probability, parents[0], parents[1])

    def doCrossover(
        self, probability: float, parent1: FloatSolution, parent2: FloatSolution
    ) -> List[FloatSolution]:
        """Perform the arithmetic crossover operation.
        
        Args:
            probability: Crossover probability
            parent1: First parent solution
            parent2: Second parent solution
            
        Returns:
            A list containing two offspring solutions
        """
        # Create copies of the parents as the base for the offspring
        offspring1 = parent1.__class__(
            parent1.lower_bound,
            parent1.upper_bound,
            len(parent1.objectives),
            len(parent1.constraints) if hasattr(parent1, 'constraints') else 0
        )
        offspring2 = parent2.__class__(
            parent2.lower_bound,
            parent2.upper_bound,
            len(parent2.objectives),
            len(parent2.constraints) if hasattr(parent2, 'constraints') else 0
        )
        
        # If crossover doesn't happen, return copies of the parents
        if random.random() >= probability:
            offspring1.variables = parent1.variables.copy()
            offspring2.variables = parent2.variables.copy()
            return [offspring1, offspring2]
        
        # Generate a single alpha for all variables in this crossover
        alpha = random.random()
        
        # Initialize variables for both offspring with the correct length
        num_variables = len(parent1.variables)
        # Initialize variables as lists
        vars1 = [0.0] * num_variables
        vars2 = [0.0] * num_variables
        
        # Perform arithmetic crossover on each variable
        for i in range(num_variables):
            p1 = parent1.variables[i]
            p2 = parent2.variables[i]
            
            # Calculate new values using the same alpha for all variables
            value1 = alpha * p1 + (1 - alpha) * p2
            value2 = (1 - alpha) * p1 + alpha * p2
            
            # Apply bounds repair if needed
            lower_bound = parent1.lower_bound[i]
            upper_bound = parent1.upper_bound[i]
            
            repaired1 = self.repair_operator(value1, lower_bound, upper_bound)
            repaired2 = self.repair_operator(value2, lower_bound, upper_bound)
            
            vars1[i] = repaired1
            vars2[i] = repaired2
            
        # Set the variables after all calculations are done
        offspring1._variables = vars1
        offspring2._variables = vars2
        
        return [offspring1, offspring2]
    
    def _default_repair(self, value: float, lower_bound: float, upper_bound: float) -> float:
        """Default repair method that clamps values to bounds."""
        return max(lower_bound, min(upper_bound, value))
    
    def get_number_of_parents(self) -> int:
        return 2
    
    def get_number_of_children(self) -> int:
        return 2
    
    def get_name(self) -> str:
        return "Arithmetic Crossover"


class UnimodalNormalDistributionCrossover(Crossover[FloatSolution, FloatSolution]):
    """Unimodal Normal Distribution Crossover (UNDX) for real-valued solutions.
    
    UNDX is a multi-parent crossover operator that generates offspring based on the normal
    distribution defined by three parent solutions. It is particularly effective for continuous
    optimization problems as it preserves the statistics of the population.
    
    Reference:
        Onikura, T., & Kobayashi, S. (1999). Extended UNIMODAL DISTRIBUTION CROSSOVER for
        REAL-CODED GENETIC ALGORITHMS. In Proceedings of the 1999 Congress on Evolutionary
        Computation-CEC99 (Cat. No. 99TH8406) (Vol. 2, pp. 1581-1588). IEEE.
    
    Args:
        probability: Crossover probability (0.0 to 1.0)
        zeta: Controls the spread along the line connecting parents (typically in [0.1, 1.0],
            where smaller values produce offspring closer to the parents)
        eta: Controls the spread in the orthogonal direction (typically in [0.1, 0.5],
            where smaller values produce more concentrated distributions)
        repair_operator: Optional function to repair out-of-bounds values.
            If None, values are clamped to the variable bounds using min/max.
            Signature: repair_operator(value: float, lower_bound: float, upper_bound: float) -> float
    
    Raises:
        ValueError: If probability is not in [0,1] or if zeta or eta are negative
    """

    def __init__(
        self,
        probability: float = 0.9,
        zeta: float = 0.5,
        eta: float = 0.35,
        repair_operator: Optional[Callable[[float, float, float], float]] = None,
    ):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if zeta < 0:
            raise ValueError("zeta must be non-negative")
        if eta < 0:
            raise ValueError("eta must be non-negative")

        super().__init__(probability=probability)
        self.zeta = zeta
        self.eta = eta
        self.repair_operator = repair_operator if repair_operator else self._default_repair

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        Check.that(len(parents) >= 3, "UNDX requires at least three parents")
        return self.doCrossover(self.probability, parents[0], parents[1], parents[2])

    def doCrossover(
        self,
        probability: float,
        parent1: FloatSolution,
        parent2: FloatSolution,
        parent3: FloatSolution,
    ) -> List[FloatSolution]:
        """Perform the UNDX crossover operation.
        
        Args:
            probability: Crossover probability
            parent1: First parent solution
            parent2: Second parent solution
            parent3: Third parent solution (used to determine the orthogonal direction)
            
        Returns:
            A list containing two offspring solutions
        """
        # Create offspring as copies of parents initially
        offspring1 = parent1.__class__(
            parent1.lower_bound,
            parent1.upper_bound,
            len(parent1.objectives),
            len(parent1.constraints) if hasattr(parent1, 'constraints') else 0
        )
        offspring2 = parent2.__class__(
            parent2.lower_bound,
            parent2.upper_bound,
            len(parent2.objectives),
            len(parent2.constraints) if hasattr(parent2, 'constraints') else 0
        )
        
        # If crossover doesn't happen, return copies of the parents
        if random.random() >= probability:
            offspring1.variables = parent1.variables.copy()
            offspring2.variables = parent2.variables.copy()
            return [offspring1, offspring2]
        
        number_of_variables = len(parent1.variables)
        
        # Calculate the center of mass between parent1 and parent2
        center = [
            (p1 + p2) / 2.0 
            for p1, p2 in zip(parent1.variables, parent2.variables)
        ]
        
        # Calculate the difference vector between parent1 and parent2
        diff = [p2 - p1 for p1, p2 in zip(parent1.variables, parent2.variables)]
        distance = math.sqrt(sum(d * d for d in diff))
        
        # If parents are too close, return exact copies to avoid division by zero
        if distance < 1e-10:
            offspring1.variables = parent1.variables.copy()
            offspring2.variables = parent2.variables.copy()
            return [offspring1, offspring2]
        
        # Generate offspring
        for i in range(number_of_variables):
            # Generate values along the line connecting the parents
            alpha = random.uniform(-self.zeta * distance, self.zeta * distance)
            
            # Generate values in the orthogonal direction
            # Calculate beta as the sum of two random values centered around 0
            beta = (random.random() - 0.5) * self.eta * distance + \
                   (random.random() - 0.5) * self.eta * distance
            
            # Calculate the orthogonal component from parent3
            orthogonal = (parent3.variables[i] - center[i]) / distance if distance > 0 else 0.0
            
            # Create the new values
            value1 = center[i] + alpha * diff[i] / distance + beta * orthogonal
            value2 = center[i] - alpha * diff[i] / distance - beta * orthogonal
            
            # Apply bounds repair if needed
            lower_bound = parent1.lower_bound[i]
            upper_bound = parent1.upper_bound[i]
            
            offspring1._variables[i] = self.repair_operator(value1, lower_bound, upper_bound)
            offspring2._variables[i] = self.repair_operator(value2, lower_bound, upper_bound)
        
        return [offspring1, offspring2]
    
    def _default_repair(self, value: float, lower_bound: float, upper_bound: float) -> float:
        """Default repair method that clamps values to bounds."""
        return max(lower_bound, min(upper_bound, value))
    
    def get_number_of_parents(self) -> int:
        return 3  # UNDX requires exactly 3 parents
    
    def get_number_of_children(self) -> int:
        return 2  # UNDX generates 2 offspring
    
    def get_name(self) -> str:
        return f"Unimodal Normal Distribution Crossover (ζ={self.zeta}, η={self.eta})"


class DifferentialEvolutionCrossover(Crossover[FloatSolution, FloatSolution]):
    """Differential Evolution (DE) crossover operator for real-valued solutions.
    
    This operator implements the standard DE crossover used in the DE/rand/1/bin and DE/best/1/bin
    variants. It creates a trial vector by combining the target vector with a difference vector,
    then performs binomial crossover between the target and trial vectors.
    
    The operator requires three parents and three mutation factors (F, CR, and K). The first parent
    is the target vector, while the other two are used to compute the difference vector.
    
    Args:
        cr: Crossover probability (0.0 to 1.0). Controls the probability of each variable being
            taken from the trial vector versus the target vector.
        f: Differential weight (mutation factor) for the difference vector. Typically in [0, 2].
        k: Scaling factor for the difference vector. Typically in [0, 1].
        
    Raises:
        ValueError: If cr is not in [0,1] or f/k are negative.
        
    Reference:
        Storn, R., & Price, K. (1997). Differential evolution - a simple and efficient heuristic for
        global optimization over continuous spaces. Journal of global optimization, 11(4), 341-359.
    """

    def __init__(self, CR: float, F: float, K: float = 0.5):
        super(DifferentialEvolutionCrossover, self).__init__(probability=1.0)
        self.CR = CR
        self.F = F
        self.K = K

        self.current_individual: FloatSolution = None

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        """Execute the differential evolution crossover ('best/1/bin' variant in jMetal)."""
        if len(parents) != self.get_number_of_parents():
            raise Exception("The number of parents is not {}: {}".format(self.get_number_of_parents(), len(parents)))

        child = copy.deepcopy(self.current_individual)

        number_of_variables = len(parents[0].variables)
        rand = random.randint(0, number_of_variables - 1)

        for i in range(number_of_variables):
            if random.random() < self.CR or i == rand:
                value = parents[2].variables[i] + self.F * (parents[0].variables[i] - parents[1].variables[i])

                if value < child.lower_bound[i]:
                    value = child.lower_bound[i]
                if value > child.upper_bound[i]:
                    value = child.upper_bound[i]
            else:
                value = child.variables[i]

            child._variables[i] = value

        return [child]

    def get_number_of_parents(self) -> int:
        return 3

    def get_number_of_children(self) -> int:
        return 1

    def get_name(self) -> str:
        return "Differential Evolution crossover"


class CompositeCrossover(Crossover[CompositeSolution, CompositeSolution]):
    __EPS = 1.0e-14

    def __init__(self, crossover_operator_list: List[Crossover]):
        super(CompositeCrossover, self).__init__(probability=1.0)

        Check.is_not_none(crossover_operator_list)
        Check.collection_is_not_empty(crossover_operator_list)

        self.crossover_operators_list = []
        for operator in crossover_operator_list:
            Check.that(issubclass(operator.__class__, Crossover), "Object is not a subclass of Crossover")
            self.crossover_operators_list.append(operator)

    def execute(self, solutions: List[CompositeSolution]) -> List[CompositeSolution]:
        Check.is_not_none(solutions)
        Check.that(len(solutions) == 2, "The number of parents is not two: " + str(len(solutions)))

        offspring1 = []
        offspring2 = []

        number_of_solutions_in_composite_solution = len(solutions[0].variables)

        for i in range(number_of_solutions_in_composite_solution):
            parents = [solutions[0].variables[i], solutions[1].variables[i]]
            children = self.crossover_operators_list[i].execute(parents)
            offspring1.append(children[0])
            offspring2.append(children[1])

        return [CompositeSolution(offspring1), CompositeSolution(offspring2)]

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "Composite crossover"
