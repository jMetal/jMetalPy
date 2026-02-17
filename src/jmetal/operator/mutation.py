import math
import random
from typing import Callable, Optional, List

import numpy as np
from jmetal.operator.repair import ensure_float_repair, ensure_integer_repair

from jmetal.core.operator import Mutation
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
.. module:: mutation
   :platform: Unix, Windows
   :synopsis: Module implementing mutation operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class NullMutation(Mutation[Solution]):
    def __init__(self):
        super(NullMutation, self).__init__(probability=0)

    def execute(self, solution: Solution) -> Solution:
        return solution

    def get_name(self) -> str:
        """Get the name of the operator.
        
        Returns:
            str: A string containing the operator name and mutation probability.
        """
        return f"Null mutation (p={self.probability})"


class BitFlipMutation(Mutation[BinarySolution]):
    """
    NumPy-optimized bit flip mutation for BinarySolution.
    
    This implementation uses NumPy's vectorized operations for better performance
    when working with BinarySolution solutions. It flips each bit with a given
    probability, but does so using efficient array operations.
    
    Args:
        probability: The probability of flipping each bit (0.0 to 1.0)
        
    Raises:
        ValueError: If probability is not in range [0.0, 1.0]
    """
    
    def __init__(self, probability: float):
        if not (0.0 <= probability <= 1.0):
            raise ValueError(f"Probability must be in range [0.0, 1.0], got {probability}")
        super(BitFlipMutation, self).__init__(probability=probability)
    
    def execute(self, solution: BinarySolution) -> BinarySolution:
        """
        Execute the bit flip mutation operation.
        
        Args:
            solution: The solution to be mutated. Must be a BinarySolution with a 'bits' attribute.
            
        Returns:
            The mutated solution (modified in-place)
            
        Raises:
            TypeError: If solution is not a BinarySolution or doesn't have a 'bits' attribute
            ValueError: If the solution has no variables or invalid bit values
            
        Note:
            The input solution is modified in-place and also returned.
        """
        # Input validation
        if not isinstance(solution, BinarySolution):
            raise TypeError(f"Expected BinarySolution, got {type(solution).__name__}")
            
        if not hasattr(solution, 'bits') or not isinstance(solution.bits, np.ndarray):
            raise AttributeError("Solution must have a 'bits' attribute of type numpy.ndarray")
            
        if solution.number_of_variables <= 0:
            raise ValueError("Solution must have at least one variable")
            
        if len(solution.bits) == 0:
            return solution  # Nothing to mutate
            
        try:
            # Generate random numbers for each bit
            rand_values = np.random.random(solution.number_of_variables)
            
            # Create a mask of bits to flip
            flip_mask = rand_values < self.probability
            
            # Ensure the mask has the same shape as solution.bits
            if flip_mask.shape != solution.bits.shape:
                flip_mask = np.resize(flip_mask, solution.bits.shape)
            
            # Flip the bits where the mask is True
            solution.bits ^= flip_mask.astype(bool)
            
            return solution
            
        except Exception as e:
            raise RuntimeError(f"Error during bit flip mutation: {str(e)}") from e
    
    def get_name(self) -> str:
        """
        Return the name of the operator.
        
        Returns:
            str: A string representing the name of the operator
        """
        return "Bit flip mutation"


class PolynomialMutation(Mutation[FloatSolution]):
    """Implementation of a polynomial mutation operator for real-valued solutions.
    
    The polynomial mutation is based on a polynomial probability distribution that
    perturbs solutions in a way that favors small changes while still allowing
    occasional larger jumps. This provides a good balance between exploration and
    exploitation in evolutionary algorithms.
    
    The mutation follows a polynomial probability distribution centered on the
    parent value, with the spread controlled by the distribution index.
    
    Args:
        probability: The probability of mutating each variable (0 ≤ p ≤ 1).
        distribution_index: Controls the perturbation magnitude (must be ≥ 0):
            - Lower values (e.g., 5-20): More exploratory, larger mutations
            - Medium values (e.g., 20-100): Balanced exploration/exploitation
            - Higher values (e.g., >100): More exploitative, smaller mutations
        repair_operator: Optional function to repair out-of-bounds values.
            If None, values are clamped to the variable bounds.
            
    Raises:
        ValueError: If probability is not in [0,1] or distribution_index is negative.
    """

    def __init__(
        self,
        probability: float = 0.01,
        distribution_index: float = 20.0,
        repair_operator: Optional[Callable[[float, float, float], float]] = None,
        rng: Optional[object] = None,
    ):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if distribution_index < 0:
            raise ValueError("distribution_index must be non-negative")
            
        super(PolynomialMutation, self).__init__(probability=probability)
        self.distribution_index = distribution_index
        # Normalize repair operator to a FloatRepairOperator instance
        self.repair_operator = ensure_float_repair(repair_operator)
        # RNG generator (np.random.Generator) for reproducibility
        self.rng = rng if rng is not None else np.random.default_rng()

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(issubclass(type(solution), FloatSolution), "Solution type invalid")
                
        for i in range(len(solution._variables)):
            if self.rng.random() <= self.probability:
                y = solution._variables[i]
                yl, yu = solution.lower_bound[i], solution.upper_bound[i]

                if yl == yu:
                    y = yl
                else:
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    rnd = self.rng.random()
                    mut_pow = 1.0 / (self.distribution_index + 1.0)
                    if rnd <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy, self.distribution_index + 1.0))
                        deltaq = pow(val, mut_pow) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (pow(xy, self.distribution_index + 1.0))
                        deltaq = 1.0 - pow(val, mut_pow)

                    y += deltaq * (yu - yl)
                    # Use repair scalar API to enforce bounds / custom repair
                    y = self.repair_operator.repair_scalar(y, yl, yu)
                    solution._variables[i] = y
        
        return solution

    def get_name(self):
        return "Polynomial mutation"


class IntegerPolynomialMutation(Mutation[IntegerSolution]):
    """Polynomial mutation operator for integer-valued decision variables.
    
    This operator adapts the polynomial mutation for integer solutions by rounding
    the continuous values to the nearest integer. It's particularly useful for
    problems where variables must take discrete integer values.
    
    The mutation works by:
    1. Applying polynomial mutation to the integer variable (treated as float)
    2. Rounding the result to the nearest integer
    3. Clamping the value to the variable's bounds
    
    Args:
        probability: The probability of mutating each variable (0 ≤ p ≤ 1).
        distribution_index: Controls the perturbation magnitude (must be ≥ 0):
            - Lower values (e.g., 5-20): More exploratory, larger mutations
            - Medium values (e.g., 20-100): Balanced exploration/exploitation
            - Higher values (e.g., >100): More exploitative, smaller mutations
            
    Example:
        >>> from jmetal.operator import IntegerPolynomialMutation
        >>> from jmetal.core.solution import IntegerSolution
        >>> 
        >>> # Create an integer solution with bounds [0, 10] for all variables
        >>> solution = IntegerSolution(3, 1, 0)  # 3 variables, 1 objective, 0 constraints
        >>> solution.variables = [5, 5, 5]
        >>> solution.lower_bound = [0] * 3
        >>> solution.upper_bound = [10] * 3
        >>> 
        >>> # Apply polynomial mutation with 100% probability
        >>> mutation = IntegerPolynomialMutation(probability=1.0, distribution_index=20.0)
        >>> mutated = mutation.execute(solution)
        >>> # Variables will be mutated with integer values within [0, 10]
    """
    def __init__(self, probability: float, distribution_index: float = 20.0,
                 repair_operator: Optional[Callable[[float, int, int], int]] = None,
                 rng: Optional[object] = None):
        super(IntegerPolynomialMutation, self).__init__(probability=probability)
        self.distribution_index = distribution_index
        # Normalize integer repair operator
        self.repair_operator = ensure_integer_repair(repair_operator)
        self.rng = rng if rng is not None else np.random.default_rng()

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        Check.that(issubclass(type(solution), IntegerSolution), "Solution type invalid")
        
        
        for i in range(len(solution._variables)):
            if self.rng.random() <= self.probability:
                y = solution._variables[i]
                yl, yu = solution.lower_bound[i], solution.upper_bound[i]

                if yl == yu:
                    y = yl
                else:
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    mut_pow = 1.0 / (self.distribution_index + 1.0)
                    rnd = self.rng.random()
                    if rnd <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (self.distribution_index + 1.0))
                        deltaq = val**mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (self.distribution_index + 1.0))
                        deltaq = 1.0 - val**mut_pow

                    y += deltaq * (yu - yl)
                    # Use integer repair operator to round and clamp
                    y = self.repair_operator.repair_scalar(y, yl, yu)
                
                solution._variables[i] = y
                    
        return solution

    def get_name(self) -> str:
        """Get the name of the operator.
        
        Returns:
            str: A string containing the operator name and distribution index.
        """
        return f"Polynomial mutation (int, η={self.distribution_index})"


class SimpleRandomMutation(Mutation[FloatSolution]):
    """Implementation of a simple random mutation operator for real-valued solutions.
    
    This operator replaces the value of a decision variable with a random value
    uniformly distributed between the lower and upper bounds of that variable.
    This is one of the simplest mutation operators but can be effective for
    exploration, especially in the early stages of optimization.
    
    The mutation works by:
    1. For each variable, with probability `probability`:
       - Replace the variable's value with a random value from a uniform distribution
         between the variable's lower and upper bounds
    2. Leave the variable unchanged otherwise
    
    Args:
        probability: The probability of mutating each variable (0 ≤ p ≤ 1).
                    Higher values increase exploration but may disrupt good solutions.
                    
    Example:
        >>> from jmetal.operator import SimpleRandomMutation
        >>> from jmetal.core.solution import FloatSolution
        >>> 
        >>> # Create a solution with bounds [0, 10] for all variables
        >>> solution = FloatSolution([0, 0], [10, 10], 1)  # 2 variables, 1 objective
        >>> solution.variables = [5.0, 5.0]  # Initial values
        >>> 
        >>> # Apply random mutation with 50% probability
        >>> mutation = SimpleRandomMutation(probability=0.5)
        >>> mutated = mutation.execute(solution)
        >>> # Each variable has a 50% chance to be replaced with a random value in [0, 10]
    
    Args:
        probability: The probability of mutating each variable (0 ≤ p ≤ 1).
            
    Raises:
        ValueError: If probability is not in [0,1].
    """
    
    def __init__(self, probability: float, rng: Optional[object] = None):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        super(SimpleRandomMutation, self).__init__(probability=probability)
        self.rng = rng if rng is not None else np.random.default_rng()

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(issubclass(type(solution), FloatSolution), "Solution type invalid")
                
        for i in range(len(solution._variables)):
            if self.rng.random() <= self.probability:
                new_value = self.rng.uniform(solution.lower_bound[i], solution.upper_bound[i])
                solution._variables[i] = new_value
        
        return solution

    def get_name(self) -> str:
        return "Simple random mutation"


class UniformMutation(Mutation[FloatSolution]):
    """Implementation of a uniform mutation operator for real-valued solutions.
    
    This operator adds a random perturbation uniformly distributed in
    [-perturbation/2, perturbation/2] to each variable with a given probability.
    The perturbation is scaled by the variable's range, making the operator
    scale-invariant to the problem's bounds.
    
    Args:
        probability: The probability of mutating each variable (0 ≤ p ≤ 1).
        perturbation: Controls the maximum relative perturbation size (must be > 0).
            - Smaller values (e.g., 0.1-0.5): Small, local perturbations
            - Larger values (e.g., 1.0-2.0): Larger, more exploratory perturbations
            
    Raises:
        ValueError: If probability is not in [0,1] or perturbation is not positive.
    """
    
    def __init__(self, probability: float, perturbation: float = 0.5,
                 repair_operator: Optional[Callable[[float, float, float], float]] = None,
                 rng: Optional[object] = None):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if perturbation <= 0:
            raise ValueError("perturbation must be positive")
            
        super(UniformMutation, self).__init__(probability=probability)
        self.perturbation = perturbation
        # Normalize repair operator and store RNG generator (np.random.Generator)
        self.repair_operator = ensure_float_repair(repair_operator)
        self.rng = rng if rng is not None else np.random.default_rng()
    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(issubclass(type(solution), FloatSolution), "Solution type invalid")
                
        # Vectorized path: compute mask of mutated indices and apply perturbations in bulk
        n = len(solution._variables)
        if n == 0:
            return solution

        # Convert to numpy arrays for vectorized operations
        vars_arr = np.asarray(solution._variables, dtype=float)
        lbs = np.asarray(solution.lower_bound, dtype=float)
        ubs = np.asarray(solution.upper_bound, dtype=float)

        # Use configured repair operator and RNG
        repair = self.repair_operator

        # Draw random mask for which variables mutate (use configured RNG)
        mask = self.rng.random(n) < self.probability
        if not mask.any():
            return solution

        # Compute perturbations for all variables then select mutated ones
        ranges = ubs - lbs
        deltas = (self.rng.random(n) - 0.5) * self.perturbation * ranges
        candidate = vars_arr + deltas
        # Keep original where not mutated
        candidate = np.where(mask, candidate, vars_arr)

        # Repair vectorized
        repaired = repair.repair_vector(candidate, lbs, ubs)
        solution._variables = repaired.tolist()
            
        return solution

    def get_name(self) -> str:
        return f"Uniform mutation"


class NonUniformMutation(Mutation[FloatSolution]):
    """Implementation of a non-uniform mutation operator for real-valued solutions.
    
    This operator perturbs solutions in a way that the mutation strength decreases
    over time, allowing for more exploration in early generations and more
    exploitation in later generations. The mutation strength is controlled by
    the current iteration number relative to the maximum number of iterations.
    
    The mutation follows the formula:
        Δ(t, y) = y * (r * (1 - t/T)^b - 1)  if r ≤ 0.5
        Δ(t, y) = y * (1 - r * (1 - t/T)^b)  if r > 0.5
    where:
    - t is the current iteration
    - T is max_iterations
    - b is the perturbation index
    - r is a random number in [0,1]
    - y is the variable's range
    
    The operator is particularly useful for:
    - Fine-tuning solutions in later generations
    - Problems requiring adaptive exploration/exploitation balance
    - Situations where solution precision increases over time
    
    Args:
        probability: The probability of mutating each variable (0 ≤ p ≤ 1).
        perturbation: Controls the perturbation strength (must be > 0).
            - Lower values (e.g., 1-5): Smoother decrease in mutation strength
            - Higher values (e.g., 5-20): Faster transition to smaller mutations
        max_iterations: The maximum number of iterations/generations (must be > 0).
            This is used to calculate the current progress (t/T).
            
    Example:
        >>> from jmetal.operator import NonUniformMutation
        >>> from jmetal.core.solution import FloatSolution
        >>> 
        >>> # Create a solution with bounds [0, 10] for all variables
        >>> solution = FloatSolution([0, 0], [10, 10], 1)  # 2 variables, 1 objective
        >>> solution.variables = [5.0, 5.0]  # Initial values
        >>> 
        >>> # Create a non-uniform mutation operator
        >>> # With 30% mutation probability, medium perturbation (5.0), and 1000 max iterations
        >>> mutation = NonUniformMutation(probability=0.3, perturbation=5.0, max_iterations=1000)
        >>> 
        >>> # In early generations (e.g., iteration 10 of 1000)
        >>> mutation.current_iteration = 10
        >>> mutated_early = mutation.execute(solution)
        >>> 
        >>> # In later generations (e.g., iteration 900 of 1000)
        >>> mutation.current_iteration = 900
        >>> mutated_late = mutation.execute(solution)
        >>> # Later mutations will be much smaller in magnitude
    
    Note:
        Remember to update `current_iteration` before each generation to ensure
        proper adaptation of the mutation strength.
        
    Raises:
        ValueError: If probability is not in [0,1] or parameters are not positive.
    """
    
    def __init__(self, probability: float, perturbation: float = 0.5, max_iterations: int = 1000,
                 repair_operator: Optional[Callable[[float, float, float], float]] = None,
                 rng: Optional[object] = None):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if perturbation <= 0:
            raise ValueError("perturbation must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
            
        super(NonUniformMutation, self).__init__(probability=probability)
        self.perturbation = perturbation
        self.max_iterations = max_iterations
        self.current_iteration = 0
        # Normalize repair operator
        self.repair_operator = ensure_float_repair(repair_operator)
        self.rng = rng if rng is not None else np.random.default_rng()

    def execute(self, solution: FloatSolution) -> FloatSolution:
        """Execute the non-uniform mutation on a solution.
        
        Args:
            solution: The solution to be mutated.
            
        Returns:
            The mutated solution.
        """
        Check.that(issubclass(type(solution), FloatSolution), "Solution type invalid")

        for i in range(len(solution.variables)):
            if self.rng.random() <= self.probability:
                current_value = solution.variables[i]

                # Calculate delta based on direction
                if self.rng.random() <= 0.5:
                    delta = self.__delta(
                        solution.upper_bound[i] - current_value,
                        self.perturbation,
                    )
                else:
                    delta = self.__delta(
                        solution.lower_bound[i] - current_value,
                        self.perturbation,
                    )

                # Apply mutation and repair
                new_value = current_value + delta
                new_value = self.repair_operator.repair_scalar(new_value, solution.lower_bound[i], solution.upper_bound[i])
                solution._variables[i] = new_value

        return solution

    def set_current_iteration(self, current_iteration: int) -> None:
        """Set the current iteration number for controlling mutation strength.
        
        Args:
            current_iteration: The current iteration number (must be ≥ 0).
        """
        if current_iteration < 0:
            raise ValueError("current_iteration must be non-negative")
        self.current_iteration = current_iteration

    def __delta(self, y: float, b_mutation_parameter: float) -> float:
        """Calculate the non-uniform mutation delta.
        
        Args:
            y: Distance to the bound
            b_mutation_parameter: Perturbation parameter
            
        Returns:
            The mutation delta value
        """
        if y == 0:
            return 0
            
        r = self.rng.random()
        iter_frac = min(self.current_iteration / self.max_iterations, 1.0)
        
        # Calculate mutation strength based on current iteration
        delta = 1 - math.pow(iter_frac, 1 / (1 + b_mutation_parameter))
        
        # Apply mutation in appropriate direction
        if r <= 0.5:
            return y * (math.pow(2 * r + (1 - 2 * r) * delta, 1 / (1 + b_mutation_parameter)) - 1)
        return y * (1 - math.pow(2 * (1 - r) + 2 * (r - 0.5) * delta, 1 / (1 + b_mutation_parameter)))

    def get_name(self) -> str:
        """Get the name of the operator.
        
        Returns:
            A string containing the operator name and parameters.
        """
        return (f"Non-Uniform mutation (perturbation={self.perturbation}, "
                f"max_iter={self.max_iterations})")


class PermutationSwapMutation(Mutation[PermutationSolution]):
    """Implementation of a swap mutation operator for permutation solutions.
    
    This operator randomly selects two distinct positions in the permutation and swaps their values.
    It is commonly used for permutation-based optimization problems like the Traveling Salesman Problem (TSP).
    
    The mutation works by:
    1. Randomly selecting two distinct positions in the permutation
    2. Swapping the values at these positions
    3. Only performing the swap with a given probability
    
    Args:
        probability: The probability of applying the mutation to a solution (0 ≤ p ≤ 1).
                    If the probability is 1.0, the mutation is always applied.
                    
    Example:
        >>> from jmetal.operator import PermutationSwapMutation
        >>> from jmetal.core.solution import PermutationSolution
        >>> 
        >>> # Create a permutation solution [0, 1, 2, 3, 4]
        >>> solution = PermutationSolution(5, 1)  # 5 variables, 1 objective
        >>> solution.variables = [0, 1, 2, 3, 4]
        >>> 
        >>> # Apply swap mutation with 100% probability
        >>> mutation = PermutationSwapMutation(probability=1.0)
        >>> mutated = mutation.execute(solution)
        >>> # Two random positions will be swapped, e.g., [2, 1, 0, 3, 4]
    """
    def __init__(self, probability: float, rng: Optional[object] = None):
        super(PermutationSwapMutation, self).__init__(probability=probability)
        self.rng = rng if rng is not None else np.random.default_rng()

    def execute(self, solution: PermutationSolution) -> PermutationSolution:
        Check.that(issubclass(type(solution), PermutationSolution), "Solution type invalid")

        if self.rng.random() <= self.probability:
            size = len(solution.variables)
            # choose two distinct indices without replacement
            idx = self.rng.choice(size, size=2, replace=False)
            pos_one, pos_two = int(idx[0]), int(idx[1])
            solution.variables[pos_one], solution.variables[pos_two] = (
                solution.variables[pos_two],
                solution.variables[pos_one],
            )

        return solution

    def get_name(self):
        return "Permutation Swap mutation"


class CompositeMutation(Mutation[Solution]):
    """A composite mutation operator that applies different mutation operators to different solution components.
    
    This operator is particularly useful for composite solutions where each component may require
    a different mutation strategy. It maintains a list of mutation operators, one for each component
    of the composite solution.
    
    The mutation works by:
    1. Taking a composite solution as input
    2. Applying each mutation operator to the corresponding solution component
    3. Combining the results into a new composite solution
    
    Args:
        mutation_operator_list: A list of mutation operators, one for each component of the composite solution.
                               The length of this list must match the number of variables in the composite solution.
                               
    Raises:
        ValueError: If the mutation_operator_list is empty or None.
        TypeError: If any element in mutation_operator_list is not a subclass of Mutation.
        
    Example:
        >>> from jmetal.operator import CompositeMutation, BitFlipMutation, PolynomialMutation
        >>> from jmetal.core.solution import CompositeSolution, BinarySolution, FloatSolution
        >>> 
        >>> # Create a composite solution with binary and float components
        >>> binary_solution = BinarySolution(5, 1)  # 5 bits, 1 objective
        >>> float_solution = FloatSolution([0]*3, [1]*3, 1)  # 3 variables, 1 objective
        >>> composite = CompositeSolution([binary_solution, float_solution])
        >>> 
        >>> # Create a composite mutation with appropriate operators for each component
        >>> mutation = CompositeMutation([
        ...     BitFlipMutation(0.1),      # For binary component
        ...     PolynomialMutation(0.1, 20)  # For float component
        ... ])
        >>> 
        >>> # Apply the composite mutation
        >>> mutated = mutation.execute(composite)
    """
    def __init__(self, mutation_operator_list: List[Mutation]):
        super(CompositeMutation, self).__init__(probability=1.0)

        Check.is_not_none(mutation_operator_list)
        Check.collection_is_not_empty(mutation_operator_list)

        self.mutation_operators_list = []
        for operator in mutation_operator_list:
            Check.that(issubclass(operator.__class__, Mutation), "Object is not a subclass of Mutation")
            self.mutation_operators_list.append(operator)

    def execute(self, solution: CompositeSolution) -> CompositeSolution:
        Check.is_not_none(solution)

        mutated_solution_components = []
        for i in range(len(solution.variables)):
            mutated_solution_components.append(self.mutation_operators_list[i].execute(solution.variables[i]))

        return CompositeSolution(mutated_solution_components)

    def get_name(self) -> str:
        """Get the name of the operator.
        
        Returns:
            str: A string containing the operator name and the names of the component operators.
        """
        operator_names = [op.get_name() for op in self.mutation_operators_list]
        return f"Composite mutation ({', '.join(operator_names)})"


class ScrambleMutation(Mutation[PermutationSolution]):
    """Implementation of a scramble mutation operator for permutation solutions.
    
    This operator selects a random subsequence of the permutation and randomly reorders
    (scrambles) the elements within that subsequence. It is particularly useful for
    permutation problems where the relative ordering of elements is important.
    
    The mutation works by:
    1. Randomly selecting a subsequence of the permutation (limited to max 20 elements)
    2. Randomly shuffling the elements within this subsequence
    3. Only performing the scramble with a given probability
    
    Args:
        probability: The probability of applying the mutation to a solution (0 ≤ p ≤ 1).
                    If the probability is 1.0, the mutation is always applied.
                    
    Example:
        >>> from jmetal.operator import ScrambleMutation
        >>> from jmetal.core.solution import PermutationSolution
        >>> 
        >>> # Create a permutation solution [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> solution = PermutationSolution(10, 1)  # 10 variables, 1 objective
        >>> solution.variables = list(range(10))
        >>> 
        >>> # Apply scramble mutation with 100% probability
        >>> mutation = ScrambleMutation(probability=1.0)
        >>> mutated = mutation.execute(solution)
        >>> # A random subsequence will be scrambled, e.g., [0, 1, 4, 3, 2, 5, 6, 7, 8, 9]
    """
    def __init__(self, probability: float, rng: Optional[object] = None):
        super(ScrambleMutation, self).__init__(probability=probability)
        self.rng = rng if rng is not None else np.random.default_rng()

    def execute(self, solution: PermutationSolution) -> PermutationSolution:
        Check.that(issubclass(type(solution), PermutationSolution), "Solution type invalid")

        if self.rng.random() <= self.probability:
            n = len(solution.variables)
            point1 = int(self.rng.integers(0, n + 1))
            point2 = int(self.rng.integers(0, n))

            if point2 >= point1:
                point2 += 1
            else:
                point1, point2 = point2, point1

            if point2 - point1 >= 20:
                point2 = point1 + 20

            values = solution.variables[point1:point2]
            # shuffle subsequence with RNG
            permuted = list(self.rng.permutation(values))
            solution.variables[point1:point2] = permuted

        return solution

    def get_name(self) -> str:
        """Get the name of the operator.
        
        Returns:
            str: A string containing the operator name.
        """
        return "Scramble mutation"


class LevyFlightMutation(Mutation[FloatSolution]):
    """Implementation of a Lévy flight mutation operator for real-valued solutions.
    
    Lévy flights are characterized by heavy-tailed distributions with infinite variance,
    producing mostly small steps with occasional very large jumps. This behavior is
    beneficial for global optimization as it provides both local search capabilities
    and the ability to escape local optima through large jumps.
    
    The implementation uses the Mantegna algorithm to generate Lévy-distributed steps:
    1. Generate u ~ Normal(0, σ_u²) where σ_u = [Γ(1+β)sin(πβ/2)/Γ((1+β)/2)β2^((β-1)/2)]^(1/β)
    2. Generate v ~ Normal(0, 1)
    3. Lévy step = u / |v|^(1/β)
    
    Args:
        mutation_probability: The probability of mutating each variable (0 ≤ p ≤ 1).
        beta: The Lévy index parameter (1 < β ≤ 2). Controls the tail heaviness:
            - Values closer to 1.0 produce heavier tails with more frequent large jumps
            - Values around 1.5 provide balanced exploration (default)
            - Values closer to 2.0 approach Gaussian behavior with fewer large jumps
        step_size: The scaling factor for Lévy steps (must be > 0). Typical values:
            - 0.001-0.01: Fine-grained local search
            - 0.01-0.05: Balance of local and global search (default: 0.01)
            - 0.05-0.1: Emphasize global exploration
        repair_operator: Optional function to repair out-of-bounds values.
            If None, values are clamped to the variable bounds.
            
    Raises:
        ValueError: If parameters are outside their valid ranges.
    """
    
    def __init__(self, mutation_probability: float = 0.01, beta: float = 1.5,
                 step_size: float = 0.01, repair_operator: Optional[Callable[[float, float, float], float]] = None,
                 rng: Optional[object] = None):
        if not 0 <= mutation_probability <= 1:
            raise ValueError("mutation_probability must be in [0, 1]")
        if not 1 < beta <= 2:
            raise ValueError("beta must be in (1, 2]")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
            
        super().__init__(probability=mutation_probability)
        self.beta = beta
        self.step_size = step_size
        # Normalize repair operator to a FloatRepairOperator instance
        self.repair_operator = ensure_float_repair(repair_operator)
        self.rng = rng if rng is not None else np.random.default_rng()

    def execute(self, solution: FloatSolution) -> FloatSolution:        
        for i in range(len(solution._variables)):
            if self.rng.random() <= self.probability:
                current_value = solution._variables[i]
                lower_bound = solution.lower_bound[i]
                upper_bound = solution.upper_bound[i]

                # Lévy flight step (Mantegna algorithm)
                levy_step = self._generate_levy_step()
                perturbation = levy_step * self.step_size * (upper_bound - lower_bound)
                new_value = current_value + perturbation

                # Repair using scalar API
                new_value = self.repair_operator.repair_scalar(new_value, lower_bound, upper_bound)
                solution._variables[i] = new_value
            
        return solution

    def _generate_levy_step(self):
        # Mantegna algorithm
        sigma_u = self._sigma_u(self.beta)
        u = self.rng.normal(0, sigma_u)
        v = self.rng.normal(0, 1)
        return u / (abs(v) ** (1 / self.beta))

    def _sigma_u(self, beta):
        import math
        numerator = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        denominator = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        return (numerator / denominator) ** (1 / beta)

    def get_name(self):
        return "Levy Flight mutation"


class PowerLawMutation(Mutation[FloatSolution]):
    """Implementation of a power-law mutation operator for real-valued solutions.
    
    The power-law distribution produces heavy-tailed perturbations that can occasionally 
    create large jumps while favoring smaller perturbations, which is beneficial for both 
    exploration and exploitation in optimization.
    
    The mutation follows the formula:
        temp_delta = rnd^(-delta)
        deltaq = 0.5 * (rnd - 0.5) * (1 - temp_delta)
        new_value = old_value + deltaq * (upper_bound - lower_bound)
    
    Args:
        probability: The probability of mutating each variable (0 ≤ p ≤ 1).
        delta: The power-law exponent parameter (must be > 0). Controls distribution shape:
            - Values < 1.0: More uniform distributions with moderate perturbations
            - Values ≈ 1.0: Balanced exploration/exploitation (default)
            - Values > 1.0: Heavy-tailed distributions favoring small perturbations
              with occasional large jumps
        repair_operator: Optional function to repair out-of-bounds values.
            If None, values are clamped to the variable bounds.
            
    Raises:
        ValueError: If probability is not in [0,1] or delta is not positive.
    """
    
    def __init__(self, probability: float = 0.01, delta: float = 1.0, 
                 repair_operator: Optional[Callable[[float, float, float], float]] = None,
                 rng: Optional[object] = None):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if delta <= 0:
            raise ValueError("delta must be positive")
            
        super().__init__(probability=probability)
        self.delta = delta
        # Normalize repair operator to a FloatRepairOperator instance
        self.repair_operator = ensure_float_repair(repair_operator if repair_operator is not None else None)
        self.rng = rng if rng is not None else np.random.default_rng()

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(issubclass(type(solution), FloatSolution), "Solution type invalid")
                
        for i in range(len(solution._variables)):
            if self.rng.random() <= self.probability:
                current_value = solution._variables[i]
                lower_bound = solution.lower_bound[i]
                upper_bound = solution.upper_bound[i]
                
                # Generate power-law perturbation
                rnd = self.rng.random()
                rnd = min(max(rnd, 1e-10), 1 - 1e-10)
                temp_delta = math.pow(rnd, -self.delta)
                deltaq = 0.5 * (rnd - 0.5) * (1 - temp_delta)
                new_value = current_value + deltaq * (upper_bound - lower_bound)
                
                # Repair using scalar API
                new_value = self.repair_operator.repair_scalar(new_value, lower_bound, upper_bound)
                
                solution._variables[i] = new_value
            
        return solution

    def _default_repair(self, value, lower, upper):
        return max(min(value, upper), lower)

    def get_name(self):
        return f"Power Law mutation (delta={self.delta})"