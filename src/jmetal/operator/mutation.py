import random
import numpy as np
import math
from typing import Callable, Optional

from jmetal.core.operator import Mutation
from jmetal.core.solution import (
    BinarySolution,
    BinarySolutionNP,
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

    def get_name(self):
        return "Null mutation"


class BitFlipMutation(Mutation[BinarySolution]):
    def __init__(self, probability: float):
        super(BitFlipMutation, self).__init__(probability=probability)

    def execute(self, solution: BinarySolution) -> BinarySolution:
        Check.that(issubclass(type(solution), BinarySolution), "Solution type invalid")

        for i in range(len(solution.variables)):
            for j in range(len(solution.variables[i])):
                rand = random.random()
                if rand <= self.probability:
                    solution.variables[i][j] = True if solution.variables[i][j] is False else False

        return solution

    def get_name(self):
        return "BitFlip mutation"


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
    ):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if distribution_index < 0:
            raise ValueError("distribution_index must be non-negative")
            
        super(PolynomialMutation, self).__init__(probability=probability)
        self.distribution_index = distribution_index
        self.repair_operator = repair_operator

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(issubclass(type(solution), FloatSolution), "Solution type invalid")
        for i in range(len(solution.variables)):
            rand = random.random()

            if rand <= self.probability:
                y = solution.variables[i]
                yl, yu = solution.lower_bound[i], solution.upper_bound[i]

                if yl == yu:
                    y = yl
                else:
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    rnd = random.random()
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
                    if y < solution.lower_bound[i]:
                        y = solution.lower_bound[i]
                    if y > solution.upper_bound[i]:
                        y = solution.upper_bound[i]

                solution.variables[i] = y

        return solution

    def get_name(self):
        return "Polynomial mutation"


class IntegerPolynomialMutation(Mutation[IntegerSolution]):
    """Polynomial mutation adapted to integer-valued decision variables.

    - probability: Per-variable mutation probability.
    - distribution_index: Controls mutation spread. Typical values ~20.0 for fine-grained moves; lower values increase exploration.
    """
    def __init__(self, probability: float, distribution_index: float = 20.0):
        super(IntegerPolynomialMutation, self).__init__(probability=probability)
        self.distribution_index = distribution_index

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        Check.that(issubclass(type(solution), IntegerSolution), "Solution type invalid")

        for i in range(len(solution.variables)):
            if random.random() <= self.probability:
                y = solution.variables[i]
                yl, yu = solution.lower_bound[i], solution.upper_bound[i]

                if yl == yu:
                    y = yl
                else:
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    mut_pow = 1.0 / (self.distribution_index + 1.0)
                    rnd = random.random()
                    if rnd <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (self.distribution_index + 1.0))
                        deltaq = val**mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (self.distribution_index + 1.0))
                        deltaq = 1.0 - val**mut_pow

                    y += deltaq * (yu - yl)
                    if y < solution.lower_bound[i]:
                        y = solution.lower_bound[i]
                    if y > solution.upper_bound[i]:
                        y = solution.upper_bound[i]

                solution.variables[i] = int(round(y))
        return solution

    def get_name(self):
        return "Polynomial mutation (Integer)"


class SimpleRandomMutation(Mutation[FloatSolution]):
    """Implementation of a simple random mutation operator for real-valued solutions.
    
    This operator replaces the value of a decision variable with a random value
    uniformly distributed between the lower and upper bounds of that variable.
    This is one of the simplest mutation operators but can be effective for
    exploration, especially in the early stages of optimization.
    
    Args:
        probability: The probability of mutating each variable (0 ≤ p ≤ 1).
            
    Raises:
        ValueError: If probability is not in [0,1].
    """
    
    def __init__(self, probability: float):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        super(SimpleRandomMutation, self).__init__(probability=probability)

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(issubclass(type(solution), FloatSolution), "Solution type invalid")
        for i in range(len(solution.variables)):
            if random.random() <= self.probability:
                solution.variables[i] = random.uniform(solution.lower_bound[i], solution.upper_bound[i])

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
    
    def __init__(self, probability: float, perturbation: float = 0.5):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if perturbation <= 0:
            raise ValueError("perturbation must be positive")
            
        super(UniformMutation, self).__init__(probability=probability)
        self.perturbation = perturbation

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(issubclass(type(solution), FloatSolution), "Solution type invalid")
        
        for i in range(len(solution.variables)):
            if random.random() <= self.probability:
                # Calculate perturbation scaled by variable range
                var_range = solution.upper_bound[i] - solution.lower_bound[i]
                delta = (random.random() - 0.5) * self.perturbation * var_range
                
                # Apply perturbation and ensure bounds
                new_value = solution.variables[i] + delta
                solution.variables[i] = max(solution.lower_bound[i], 
                                         min(solution.upper_bound[i], new_value))

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
    where t is the current iteration, T is max_iterations, and b is the perturbation.
    
    Args:
        probability: The probability of mutating each variable (0 ≤ p ≤ 1).
        perturbation: Controls the perturbation strength (must be > 0).
            - Lower values: More gradual decrease in mutation strength
            - Higher values: More rapid decrease in mutation strength
        max_iterations: The maximum number of iterations (must be > 0).
            Used to normalize the current iteration count.
            
    Raises:
        ValueError: If probability is not in [0,1] or parameters are not positive.
    """
    
    def __init__(self, probability: float, perturbation: float = 0.5, max_iterations: int = 1000):
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

    def execute(self, solution: FloatSolution) -> FloatSolution:
        """Execute the non-uniform mutation on a solution.
        
        Args:
            solution: The solution to be mutated.
            
        Returns:
            The mutated solution.
        """
        Check.that(issubclass(type(solution), FloatSolution), "Solution type invalid")

        for i in range(len(solution.variables)):
            if random.random() <= self.probability:
                current_value = solution.variables[i]
                
                # Calculate delta based on direction
                if random.random() <= 0.5:
                    delta = self.__delta(
                        solution.upper_bound[i] - current_value,
                        self.perturbation,
                    )
                else:
                    delta = self.__delta(
                        solution.lower_bound[i] - current_value,
                        self.perturbation,
                    )

                # Apply mutation and ensure bounds
                new_value = current_value + delta
                solution.variables[i] = max(solution.lower_bound[i],
                                         min(solution.upper_bound[i], new_value))

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
            
        r = random.random()
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
    def execute(self, solution: PermutationSolution) -> PermutationSolution:
        Check.that(issubclass(type(solution), PermutationSolution), "Solution type invalid")

        rand = random.random()

        if rand <= self.probability:
            pos_one, pos_two = random.sample(range(len(solution.variables)), 2)
            solution.variables[pos_one], solution.variables[pos_two] = (
                solution.variables[pos_two],
                solution.variables[pos_one],
            )

        return solution

    def get_name(self):
        return "Permutation Swap mutation"


class CompositeMutation(Mutation[Solution]):
    def __init__(self, mutation_operator_list: [Mutation]):
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
        return "Composite mutation operator"


class BitFlipNPMutation(Mutation[BinarySolutionNP]):
    """
    NumPy-optimized bit flip mutation for BinarySolutionNP.
    
    This implementation uses NumPy's vectorized operations for better performance
    when working with BinarySolutionNP solutions. It flips each bit with a given
    probability, but does so using efficient array operations.
    
    Args:
        probability: The probability of flipping each bit (0.0 to 1.0)
    """
    
    def __init__(self, probability: float):
        super(BitFlipNPMutation, self).__init__(probability=probability)
    
    def execute(self, solution: BinarySolutionNP) -> BinarySolutionNP:
        """
        Execute the bit flip mutation operation.
        
        Args:
            solution: The solution to be mutated
            
        Returns:
            The mutated solution
            
        Note:
            The input solution is modified in-place and also returned.
        """
        # Generate random numbers for each bit
        rand_values = np.random.random(solution.number_of_variables)
        
        # Create a mask of bits to flip
        flip_mask = rand_values < self.probability
        
        # Flip the bits where the mask is True
        solution.bits ^= flip_mask
        
        return solution
    
    def get_name(self) -> str:
        """Return the name of the operator."""
        return "Bit flip mutation (NP-optimized)"


class ScrambleMutation(Mutation[PermutationSolution]):
    def execute(self, solution: PermutationSolution) -> PermutationSolution:
        Check.that(issubclass(type(solution), PermutationSolution), "Solution type invalid")
        rand = random.random()

        if rand <= self.probability:
            point1 = random.randint(0, len(solution.variables))
            point2 = random.randint(0, len(solution.variables) - 1)

            if point2 >= point1:
                point2 += 1
            else:
                point1, point2 = point2, point1

            if point2 - point1 >= 20:
                point2 = point1 + 20

            values = solution.variables[point1:point2]
            solution.variables[point1:point2] = random.sample(values, len(values))

        return solution

    def get_name(self):
        return "Scramble"


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
                 step_size: float = 0.01, repair_operator = None):
        if not 0 <= mutation_probability <= 1:
            raise ValueError("mutation_probability must be in [0, 1]")
        if not 1 < beta <= 2:
            raise ValueError("beta must be in (1, 2]")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
            
        super().__init__(probability=mutation_probability)
        self.beta = beta
        self.step_size = step_size
        self.repair_operator = repair_operator

    def execute(self, solution: FloatSolution) -> FloatSolution:
        for i in range(len(solution.variables)):
            if np.random.rand() <= self.probability:
                current_value = solution.variables[i]
                lower_bound = solution.lower_bound[i]
                upper_bound = solution.upper_bound[i]

                # Lévy flight step (Mantegna algorithm)
                levy_step = self._generate_levy_step()
                perturbation = levy_step * self.step_size * (upper_bound - lower_bound)
                new_value = current_value + perturbation

                # Repair if out of bounds
                if self.repair_operator:
                    new_value = self.repair_operator(new_value, lower_bound, upper_bound)
                else:
                    new_value = min(max(new_value, lower_bound), upper_bound)

                solution.variables[i] = new_value
        return solution

    def _generate_levy_step(self):
        # Mantegna algorithm
        sigma_u = self._sigma_u(self.beta)
        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, 1)
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
                 repair_operator: Optional[Callable[[float, float, float], float]] = None):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if delta <= 0:
            raise ValueError("delta must be positive")
            
        super().__init__(probability=probability)
        self.delta = delta
        self.repair_operator = repair_operator if repair_operator is not None else self._default_repair

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(issubclass(type(solution), FloatSolution), "Solution type invalid")
        for i in range(len(solution.variables)):
            if random.random() <= self.probability:
                current_value = solution.variables[i]
                lower_bound = solution.lower_bound[i]
                upper_bound = solution.upper_bound[i]
                rnd = random.random()
                rnd = min(max(rnd, 1e-10), 1 - 1e-10)
                temp_delta = math.pow(rnd, -self.delta)
                deltaq = 0.5 * (rnd - 0.5) * (1 - temp_delta)
                new_value = current_value + deltaq * (upper_bound - lower_bound)
                new_value = self.repair_operator(new_value, lower_bound, upper_bound)
                solution.variables[i] = new_value
        return solution

    def _default_repair(self, value, lower, upper):
        return max(min(value, upper), lower)

    def get_name(self):
        return f"Power Law mutation (delta={self.delta})"