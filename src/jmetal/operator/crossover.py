import copy
import math
import random
import numpy as np
from typing import List

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

"""
.. module:: crossover
   :platform: Unix, Windows
   :synopsis: Module implementing crossover operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class NullCrossover(Crossover[Solution, Solution]):
    def __init__(self):
        super(NullCrossover, self).__init__(probability=0.0)

    def execute(self, parents: List[Solution]) -> List[Solution]:
        if len(parents) != 2:
            raise Exception("The number of parents is not two: {}".format(len(parents)))

        return parents

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "Null crossover"


class PMXCrossover(Crossover[PermutationSolution, PermutationSolution]):
    def __init__(self, probability: float):
        super(PMXCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[PermutationSolution]) -> List[PermutationSolution]:
        if len(parents) != 2:
            raise Exception("The number of parents is not two: {}".format(len(parents)))

        offspring = copy.deepcopy(parents)
        permutation_length = len(offspring[0].variables)

        rand = random.random()
        if rand <= self.probability:
            cross_points = sorted([random.randint(0, permutation_length) for _ in range(2)])

            def _repeated(element, collection):
                c = 0
                for e in collection:
                    if e == element:
                        c += 1
                return c > 1

            def _swap(data_a, data_b, cross_points):
                c1, c2 = cross_points
                new_a = data_a[:c1] + data_b[c1:c2] + data_a[c2:]
                new_b = data_b[:c1] + data_a[c1:c2] + data_b[c2:]
                return new_a, new_b

            def _map(swapped, cross_points):
                n = len(swapped[0])
                c1, c2 = cross_points
                s1, s2 = swapped
                map_ = s1[c1:c2], s2[c1:c2]
                for i_chromosome in range(n):
                    if not c1 < i_chromosome < c2:
                        for i_son in range(2):
                            while _repeated(swapped[i_son][i_chromosome], swapped[i_son]):
                                map_index = map_[i_son].index(swapped[i_son][i_chromosome])
                                swapped[i_son][i_chromosome] = map_[1 - i_son][map_index]
                return s1, s2

            swapped = _swap(offspring[0].variables, offspring[1].variables, cross_points)
            mapped = _map(swapped, cross_points)

            offspring[0].variables, offspring[1].variables = mapped

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "Partially Matched crossover"


class CXCrossover(Crossover[PermutationSolution, PermutationSolution]):
    def __init__(self, probability: float):
        super(CXCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[PermutationSolution]) -> List[PermutationSolution]:
        if len(parents) != 2:
            raise Exception("The number of parents is not two: {}".format(len(parents)))

        offspring = copy.deepcopy(parents[::-1])
        rand = random.random()

        if rand <= self.probability:
            idx = random.randint(0, len(parents[0].variables) - 1)
            curr_idx = idx
            cycle = []

            while True:
                cycle.append(curr_idx)
                curr_idx = parents[0].variables.index(parents[1].variables[curr_idx])

                if curr_idx == idx:
                    break

            for j in range(len(parents[0].variables)):
                if j in cycle:
                    offspring[0].variables[j] = parents[0].variables[j]
                    offspring[1].variables[j] = parents[1].variables[j]

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "Cycle crossover"


class SBXCrossover(Crossover[FloatSolution, FloatSolution]):
    """Simulated Binary Crossover (SBX) for real-valued solutions.
    
    SBX is a popular crossover operator for real-coded genetic algorithms that simulates the behavior of the single-point
    crossover operator in binary-coded GAs. It creates offspring solutions based on a probability distribution centered
    around the parent solutions, with the spread of the distribution controlled by the distribution index.
    
    Args:
        probability: Crossover probability (0.0 to 1.0)
        distribution_index: Distribution index (must be ≥ 0). Higher values produce offspring closer to parents.
            Typical values range from 5 to 30, with 20 being a common default.
            
    Raises:
        Exception: If distribution_index is negative
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

                        lower_bound, upper_bound = parents[0].lower_bound[i], parents[1].upper_bound[i]

                        beta = 1.0 + (2.0 * (y1 - lower_bound) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        rand = random.random()
                        if rand <= (1.0 / alpha):
                            betaq = pow(rand * alpha, (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))
                        beta = 1.0 + (2.0 * (upper_bound - y2) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        if rand <= (1.0 / alpha):
                            betaq = pow((rand * alpha), (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))

                        if c1 < lower_bound:
                            c1 = lower_bound
                        if c2 < lower_bound:
                            c2 = lower_bound
                        if c1 > upper_bound:
                            c1 = upper_bound
                        if c2 > upper_bound:
                            c2 = upper_bound

                        if random.random() <= 0.5:
                            offspring[0].variables[i] = c2
                            offspring[1].variables[i] = c1
                        else:
                            offspring[0].variables[i] = c1
                            offspring[1].variables[i] = c2
                    else:
                        offspring[0].variables[i] = value_x1
                        offspring[1].variables[i] = value_x2
                else:
                    offspring[0].variables[i] = value_x1
                    offspring[1].variables[i] = value_x2
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

                        lower_bound, upper_bound = parents[0].lower_bound[i], parents[1].upper_bound[i]

                        beta = 1.0 + (2.0 * (y1 - lower_bound) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        rand = random.random()
                        if rand <= (1.0 / alpha):
                            betaq = pow(rand * alpha, (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))
                        beta = 1.0 + (2.0 * (upper_bound - y2) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        if rand <= (1.0 / alpha):
                            betaq = pow((rand * alpha), (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))

                        if c1 < lower_bound:
                            c1 = lower_bound
                        if c2 < lower_bound:
                            c2 = lower_bound
                        if c1 > upper_bound:
                            c1 = upper_bound
                        if c2 > upper_bound:
                            c2 = upper_bound

                        if random.random() <= 0.5:
                            offspring[0].variables[i] = int(c2)
                            offspring[1].variables[i] = int(c1)
                        else:
                            offspring[0].variables[i] = int(c1)
                            offspring[1].variables[i] = int(c2)
                    else:
                        offspring[0].variables[i] = value_x1
                        offspring[1].variables[i] = value_x2
                else:
                    offspring[0].variables[i] = value_x1
                    offspring[1].variables[i] = value_x2
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
            
            # Perform the crossover by swapping bits after the crossover point
            temp = bits1[crossover_point:].copy()
            offspring[0].bits = np.concatenate([bits1[:crossover_point], bits2[crossover_point:]])
            offspring[1].bits = np.concatenate([bits2[:crossover_point], temp])
        
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


from typing import List, TypeVar, Callable, Optional
import random
from jmetal.core.operator import Crossover
from jmetal.core.solution import FloatSolution
from jmetal.util.ckecking import Check

S = TypeVar('S', bound=FloatSolution)

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
            parent2.upper_bound, 
            len(parent1.objectives),
            len(parent1.constraints) if hasattr(parent1, 'constraints') else 0
        )
        offspring2 = parent2.__class__(
            parent1.lower_bound, 
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

            offspring1.variables[i] = y1
            offspring2.variables[i] = y2

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
            parent1.lower_bound, 
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

            offspring1.variables[i] = y1
            offspring2.variables[i] = y2

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
        
        # Perform arithmetic crossover on each variable
        for i in range(len(parent1.variables)):
            p1 = parent1.variables[i]
            p2 = parent2.variables[i]
            
            # Generate a random weight for this variable
            alpha = random.random()
            
            # Calculate new values using arithmetic crossover
            value1 = alpha * p1 + (1 - alpha) * p2
            value2 = (1 - alpha) * p1 + alpha * p2
            
            # Apply bounds repair if needed
            lower_bound = parent1.lower_bound[i]
            upper_bound = parent1.upper_bound[i]
            
            offspring1.variables[i] = self.repair_operator(value1, lower_bound, upper_bound)
            offspring2.variables[i] = self.repair_operator(value2, lower_bound, upper_bound)
        
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
            
            offspring1.variables[i] = self.repair_operator(value1, lower_bound, upper_bound)
            offspring2.variables[i] = self.repair_operator(value2, lower_bound, upper_bound)
        
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

            child.variables[i] = value

        return [child]

    def get_number_of_parents(self) -> int:
        return 3

    def get_number_of_children(self) -> int:
        return 1

    def get_name(self) -> str:
        return "Differential Evolution crossover"


class CompositeCrossover(Crossover[CompositeSolution, CompositeSolution]):
    __EPS = 1.0e-14

    def __init__(self, crossover_operator_list: [Crossover]):
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
