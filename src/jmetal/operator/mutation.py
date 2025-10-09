import random
import numpy as np
import math
from typing import Callable, Optional

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
    """Polynomial mutation for real-valued decision variables.

    - probability: Per-variable mutation probability.
    - distribution_index: Controls mutation spread. Typical values ~20.0 for fine-grained moves; lower values increase exploration.
    """
    def __init__(self, probability: float, distribution_index: float = 20.0):
        super(PolynomialMutation, self).__init__(probability=probability)
        self.distribution_index = distribution_index

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
    def __init__(self, probability: float):
        super(SimpleRandomMutation, self).__init__(probability=probability)

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(issubclass(type(solution), FloatSolution), "Solution type invalid")

        for i in range(len(solution.variables)):
            rand = random.random()
            if rand <= self.probability:
                solution.variables[i] = (
                    solution.lower_bound[i] + (solution.upper_bound[i] - solution.lower_bound[i]) * random.random()
                )
        return solution

    def get_name(self):
        return "Simple random_search mutation"


class UniformMutation(Mutation[FloatSolution]):
    def __init__(self, probability: float, perturbation: float = 0.5):
        super(UniformMutation, self).__init__(probability=probability)
        self.perturbation = perturbation

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(issubclass(type(solution), FloatSolution), "Solution type invalid")

        for i in range(len(solution.variables)):
            rand = random.random()

            if rand <= self.probability:
                tmp = (random.random() - 0.5) * self.perturbation
                tmp += solution.variables[i]

                if tmp < solution.lower_bound[i]:
                    tmp = solution.lower_bound[i]
                elif tmp > solution.upper_bound[i]:
                    tmp = solution.upper_bound[i]

                solution.variables[i] = tmp

        return solution

    def get_name(self):
        return "Uniform mutation"


class NonUniformMutation(Mutation[FloatSolution]):
    def __init__(self, probability: float, perturbation: float = 0.5, max_iterations: int = 0.5):
        super(NonUniformMutation, self).__init__(probability=probability)
        self.perturbation = perturbation
        self.max_iterations = max_iterations
        self.current_iteration = 0

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(issubclass(type(solution), FloatSolution), "Solution type invalid")

        for i in range(len(solution.variables)):
            if random.random() <= self.probability:
                rand = random.random()

                if rand <= 0.5:
                    tmp = self.__delta(solution.upper_bound[i] - solution.variables[i], self.perturbation)
                else:
                    tmp = self.__delta(solution.lower_bound[i] - solution.variables[i], self.perturbation)

                tmp += solution.variables[i]

                if tmp < solution.lower_bound[i]:
                    tmp = solution.lower_bound[i]
                elif tmp > solution.upper_bound[i]:
                    tmp = solution.upper_bound[i]

                solution.variables[i] = tmp

        return solution

    def set_current_iteration(self, current_iteration: int):
        self.current_iteration = current_iteration

    def __delta(self, y: float, b_mutation_parameter: float):
        return y * (
            1.0
            - pow(
                random.random(), pow((1.0 - 1.0 * self.current_iteration / self.max_iterations), b_mutation_parameter)
            )
        )

    def get_name(self):
        return "Non-uniform mutation"


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
    def __init__(self, mutation_probability=0.01, beta=1.5, step_size=0.01, repair_operator=None):
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
    def __init__(self, probability: float = 0.01, delta: float = 1.0, repair_operator: Optional[Callable[[float, float, float], float]] = None):
        super().__init__(probability=probability)
        if delta <= 0:
            raise ValueError("Delta parameter must be positive")
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