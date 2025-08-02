import math
from math import ceil, fabs, copysign

import numpy as np

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class CONV2(FloatProblem):
    def __init__(self):
        super().__init__()

        # Problem has 2 decision variables
        self._number_of_variables = 2
        # Problem has 2 objective functions
        self._number_of_objectives = 2
        self._number_of_constraints = 0

        # Bounds for each variable: [0.0, 10.0]
        self.lower_bound = [0.0, 0.0]
        self.upper_bound = [10.0, 10.0]

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f1', 'f2']

    # Evaluates a solution by computing its two objectives
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = solution.variables[0]
        x2 = solution.variables[1]

        f1 = x1 ** 2 + x2 ** 2
        f2 = (x1 - 10) ** 2 + x2 ** 2

        solution.objectives[0] = f1
        solution.objectives[1] = f2
        return solution

    def name(self) -> str:
        return 'CONV2'

    # Accessor methods for problem metadata
    def number_of_variables(self):
        return self._number_of_variables

    def number_of_objectives(self):
        return self._number_of_objectives

    def number_of_constraints(self):
        return self._number_of_constraints


class CONV3_4(FloatProblem):
    def __init__(self):
        super().__init__()
        # Problem has 3 variables, 3 objectives, and no constraints
        self._number_of_variables = 3
        self._number_of_objectives = 3
        self._number_of_constraints = 0

        # Variables are bounded in [-3, 3]
        self.lower_bound = [-3.0] * 3
        self.upper_bound = [3.0] * 3

        self.obj_directions = [self.MINIMIZE] * 3
        self.obj_labels = ['f1', 'f2', 'f3']

        self.a1 = [-1.0, -1.0, -1.0]
        self.a2 = [1.0, 1.0, 1.0]
        self.a3 = [-1.0, 1.0, -1.0]

    # Evaluate a solution by computing three non-symmetric objectives
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables

        f1 = (x[0] - self.a1[0]) ** 4 + (x[1] - self.a1[1]) ** 2 + (x[2] - self.a1[2]) ** 2
        f2 = (x[0] - self.a2[0]) ** 2 + (x[1] - self.a2[1]) ** 4 + (x[2] - self.a2[2]) ** 2
        f3 = (x[0] - self.a3[0]) ** 2 + (x[1] - self.a3[1]) ** 2 + (x[2] - self.a3[2]) ** 4

        solution.objectives[0] = f1
        solution.objectives[1] = f2
        solution.objectives[2] = f3

        return solution

    def name(self) -> str:
        return 'CONV3_4'

    # Accessor methods
    def number_of_variables(self): return self._number_of_variables

    def number_of_objectives(self): return self._number_of_objectives

    def number_of_constraints(self): return self._number_of_constraints


class CONV3(FloatProblem):
    def __init__(self):
        super().__init__()

        # The problem has 3 decision variables
        self._number_of_variables = 3
        # The problem has 3 objectives
        self._number_of_objectives = 3
        self._number_of_constraints = 0

        # Each variable is bounded in [-3.0, 3.0]
        self.lower_bound = [-3.0, -3.0, -3.0]
        self.upper_bound = [3.0, 3.0, 3.0]

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f1', 'f2', 'f3']

    # Compute the three objective values for a solution
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables

        # Reference points in 3D space
        a1 = [-1, -1, -1]
        a2 = [1, 1, 1]
        a3 = [-1, 1, -1]

        f1 = sum((xi - ai) ** 2 for xi, ai in zip(x, a1))
        f2 = sum((xi - ai) ** 2 for xi, ai in zip(x, a2))
        f3 = sum((xi - ai) ** 2 for xi, ai in zip(x, a3))

        solution.objectives[0] = f1
        solution.objectives[1] = f2
        solution.objectives[2] = f3

        return solution

    def name(self) -> str:
        return 'CONV3'

    # Accessor methods for problem properties
    def number_of_variables(self):
        return self._number_of_variables

    def number_of_objectives(self):
        return self._number_of_objectives

    def number_of_constraints(self):
        return self._number_of_constraints


class CONV4_2F(FloatProblem):
    def __init__(self):
        super().__init__()
        self._number_of_variables = 4
        self._number_of_objectives = 4
        self._number_of_constraints = 0

        # Variables are bounded in [-3, 3]
        self.lower_bound = [-3.0] * 4
        self.upper_bound = [3.0] * 4

        self.obj_directions = [self.MINIMIZE] * 4
        self.obj_labels = ['f1', 'f2', 'f3', 'f4']

        # Canonical basis vectors in R^4
        self.A = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.ones_vec = np.ones(4)

        a1 = self.A[0]
        a4 = self.A[3]

        phi1 = np.array([
            0,
            np.linalg.norm(a1 - self.A[1]) ** 2,
            np.linalg.norm(a1 - self.A[2]) ** 2,
            np.linalg.norm(a1 - self.A[3]) ** 2
        ])
        phi4 = np.array([
            np.linalg.norm(a4 - self.A[0]) ** 2,
            np.linalg.norm(a4 - self.A[1]) ** 2,
            np.linalg.norm(a4 - self.A[2]) ** 2,
            0
        ])

        self.sigma = phi4 - phi1

    # Evaluation: changes formula based on sign of x
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = np.array(solution.variables)
        f = np.zeros(4)

        if np.all(x < 0):
            # If all x_i < 0, compute convex formulation
            for j in range(4):
                diff = x + self.ones_vec - self.A[j]
                f[j] = np.sum(diff ** 2) - 3.5 * self.sigma[j]
        else:
            # Otherwise, compute product-based formulation
            for j in range(4):
                prod = x * self.A[j]
                f[j] = np.sum(prod ** 2)

        # Assign objectives to solution
        for j in range(4):
            solution.objectives[j] = f[j]

        return solution

    def name(self) -> str:
        return 'CONV4-2F'

    def number_of_variables(self):
        return self._number_of_variables

    def number_of_objectives(self):
        return self._number_of_objectives

    def number_of_constraints(self):
        return self._number_of_constraints


class DENT(FloatProblem):
    def __init__(self):
        super().__init__()

        # The problem has 2 decision variables
        self._number_of_variables = 2
        # The problem has 2 objectives
        self._number_of_objectives = 2
        self._number_of_constraints = 0

        # Variable bounds: both x1 and x2 in [-2.0, 2.0]
        self.lower_bound = [-2.0, -2.0]
        self.upper_bound = [2.0, 2.0]

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f1', 'f2']

    # Evaluate the solution by computing the two objectives
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = solution.variables[0]
        x2 = solution.variables[1]
        alpha = 0.85  # Parameter for exponential term

        term1 = math.sqrt(1 + (x1 + x2) ** 2)
        term2 = math.sqrt(1 + (x1 - x2) ** 2)
        exp_term = alpha * math.exp(-(x1 - x2) ** 2)

        f1 = 0.5 * (term1 + term2 + x1 - x2) + exp_term
        f2 = 0.5 * (term1 + term2 - x1 + x2) + exp_term

        solution.objectives[0] = f1
        solution.objectives[1] = f2
        return solution

    def name(self) -> str:
        return 'DENT'

    # Accessor methods
    def number_of_variables(self):
        return self._number_of_variables

    def number_of_objectives(self):
        return self._number_of_objectives

    def number_of_constraints(self):
        return self._number_of_constraints


class SYM_PART(FloatProblem):
    def __init__(self):
        super().__init__()
        # Problem has 2 decision variables
        self._number_of_variables = 2
        # Problem has 2 objectives
        self._number_of_objectives = 2
        self._number_of_constraints = 0

        # Variable bounds in [-0.5, 0.5]
        self.lower_bound = [-0.5, -0.5]
        self.upper_bound = [0.5, 0.5]

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f1', 'f2']

    # Evaluate the solution using discrete shifting rules
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        a = 0.5
        b = 5
        c = 5

        x1 = solution.variables[0]
        x2 = solution.variables[1]

        t1 = copysign(1, x1) * min(ceil((fabs(x1) - a - (c / 2)) / (2 * a + c)), 1)
        t2 = copysign(1, x2) * min(ceil((fabs(x2) - b / 2) / b), 1)

        f1 = (x1 - t1 * (c + 2 * a) + a) ** 2 + (x2 - t2 * b) ** 2
        f2 = (x1 - t1 * (c + 2 * a) - a) ** 2 + (x2 - t2 * b) ** 2

        solution.objectives[0] = f1
        solution.objectives[1] = f2

        return solution

    def name(self) -> str:
        return 'SYM-PART'

    # Accessor methods
    def number_of_variables(self):
        return self._number_of_variables

    def number_of_objectives(self):
        return self._number_of_objectives

    def number_of_constraints(self):
        return self._number_of_constraints

class SSW(FloatProblem):
    def __init__(self):
        super().__init__()

        # Fixed number of decision variables (n = 3)
        self._number_of_variables = 3
        self._number_of_objectives = 2
        self._number_of_constraints = 0

        # Search space: [0, 40]^3
        self.lower_bound = [0.0] * 3
        self.upper_bound = [40.0] * 3

        # Objective directions: minimization
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f1', 'f2']

    # Evaluate a single solution using the SSW formulation
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables

        f1 = sum(x)

        product_term = 1.0
        for j in range(len(x)):
            if j in [0, 1]:
                wj = 0.01 * math.exp(-(x[j] / 20) ** 2.5)
            else:
                wj = 0.01 * math.exp(-x[j] / 15)
            product_term *= (1 - wj)

        f2 = 1 - product_term

        solution.objectives[0] = f1
        solution.objectives[1] = f2
        return solution

    def name(self) -> str:
        return 'SSW'

    # Accessor methods
    def number_of_variables(self):
        return self._number_of_variables

    def number_of_objectives(self):
        return self._number_of_objectives

    def number_of_constraints(self):
        return self._number_of_constraints


class TWO_ON_ONE(FloatProblem):
    def __init__(self):
        super().__init__()

        # Fixed number of decision variables (n = 2)
        self._number_of_variables = 2
        self._number_of_objectives = 2
        self._number_of_constraints = 0

        # Search space: [-3, 3]^2
        self.lower_bound = [-3.0] * 2
        self.upper_bound = [3.0] * 2

        # Objective directions: minimization
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f1', 'f2']

    # Evaluate a single solution using the TWO_ON_ONE formulation
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables

        k = 0
        l = 1
        c = 10
        d = 0.25

        x1 = x[0]
        x2 = x[1]


        f1 = x1**4 + x2**4 - x1**2 + x2**2 - c * x1 * x2 + d * x1 + 20
        f2 = (x1 - k)**2 + (x2 - l)**2

        solution.objectives[0] = f1
        solution.objectives[1] = f2
        return solution

    def name(self) -> str:
        return 'TWO_ON_ONE'

    # Accessor methods
    def number_of_variables(self):
        return self._number_of_variables

    def number_of_objectives(self):
        return self._number_of_objectives

    def number_of_constraints(self):
        return self._number_of_constraints


class OMNI_TEST(FloatProblem):
    def __init__(self, number_of_variables: int = 3):
        super().__init__()

        # Number of decision variables (default 3, can be changed)
        self._number_of_variables = number_of_variables
        self._number_of_objectives = 2
        self._number_of_constraints = 0

        # Search space: [0, 6]^n
        self.lower_bound = [0.0] * number_of_variables
        self.upper_bound = [6.0] * number_of_variables

        # Objective directions: minimization
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f1', 'f2']

    # Evaluate a single solution using the OMNI-TEST formulation
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables


        f1 = sum(math.sin(math.pi * xi) for xi in x)
        f2 = sum(math.cos(math.pi * xi) for xi in x)

        solution.objectives[0] = f1
        solution.objectives[1] = f2
        return solution

    def name(self) -> str:
        return 'OMNI_TEST'

    # Accessor methods
    def number_of_variables(self):
        return self._number_of_variables

    def number_of_objectives(self):
        return self._number_of_objectives

    def number_of_constraints(self):
        return self._number_of_constraints
