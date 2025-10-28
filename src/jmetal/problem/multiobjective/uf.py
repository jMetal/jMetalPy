from math import pi, sin, cos, sqrt, exp, e
from typing import List

import numpy as np

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

"""
.. module:: UF
   :platform: Unix, Windows, macOS
   :synopsis: Problems of the CEC2009 multi-objective competition

.. moduleauthor:: Antonio J. Nebro <ajnebro@uma.es>
"""

class UF1(FloatProblem):
    """Problem UF1.

    .. note:: Unconstrained problem. The default number of variables is 30.
    """

    def __init__(self, number_of_variables: int = 30):
        """:param number_of_variables: number of decision variables of the problem."""
        super(UF1, self).__init__()
        self.lower_bound = number_of_variables * [-1.0]
        self.upper_bound = number_of_variables * [1.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0
        
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f1", "f2"]
    
    def number_of_variables(self) -> int:
        return len(self.lower_bound)
        
    def number_of_objectives(self) -> int:
        return len(self.obj_directions)
        
    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        sum1 = 0
        sum2 = 0
        count1 = 0
        count2 = 0

        x = solution.variables
        n = self.number_of_variables()

        for i in range(2, n):
            y = x[i] - sin(6.0 * pi * x[0] + (i + 1) * pi / n)
            y = y * y

            if (i + 1) % 2 == 1:  # odd indices (3, 5, 7, ...)
                sum1 += y
                count1 += 1
            else:  # even indices (4, 6, 8, ...)
                sum2 += y
                count2 += 1

        solution.objectives[0] = x[0] + 2.0 * sum1 / count1 if count1 > 0 else x[0]
        solution.objectives[1] = 1.0 - sqrt(x[0]) + 2.0 * sum2 / count2 if count2 > 0 else 1.0 - sqrt(x[0])

        return solution

    def name(self):
        return "UF1"


class UF2(FloatProblem):
    """Problem UF2.

    .. note:: Unconstrained problem. The default number of variables is 30.
    """

    def __init__(self, number_of_variables: int = 30):
        """:param number_of_variables: number of decision variables of the problem."""
        super(UF2, self).__init__()
        self.lower_bound = [0.0] + [-1.0] * (number_of_variables - 1)
        self.upper_bound = [1.0] * number_of_variables
        
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f1", "f2"]
    
    def number_of_variables(self) -> int:
        return len(self.lower_bound)
        
    def number_of_objectives(self) -> int:
        return len(self.obj_directions)
        
    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        sum1 = 0.0
        sum2 = 0.0
        n_vars = self.number_of_variables()
        for i in range(2, n_vars):
            y_base = 0.3 * x[0] * (x[0] * np.cos(24.0 * np.pi * x[0] + 4.0 * (i + 1) * np.pi / n_vars) + 2.0) * np.sin(6.0 * np.pi * x[0] + (i + 1) * np.pi / n_vars)
            y = x[i] - y_base
            if (i + 1) % 2 == 1:  # odd indices (3, 5, 7, ...)
                sum1 += y * y
            else:  # even indices (4, 6, 8, ...)
                sum2 += y * y
        solution.objectives[0] = x[0] + 2.0 * sum1 / np.floor(n_vars / 2.0)
        solution.objectives[1] = 1.0 - np.sqrt(x[0]) + 2.0 * sum2 / (np.ceil(n_vars / 2.0) - 1)
        return solution

    def name(self):
        return "UF2"


class UF3(FloatProblem):
    """Problem UF3.

    .. note:: Unconstrained problem. The default number of variables is 30.
    """

    def __init__(self, number_of_variables: int = 30):
        """:param number_of_variables: number of decision variables of the problem."""
        super(UF3, self).__init__()
        self.lower_bound = [0.0] + [-1.0] * (number_of_variables - 1)
        self.upper_bound = [1.0] * number_of_variables
        
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f1", "f2"]
    
    def number_of_variables(self) -> int:
        return len(self.lower_bound)
        
    def number_of_objectives(self) -> int:
        return len(self.obj_directions)
        
    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        sum1 = 0.0
        sum2 = 0.0
        prod1 = 1.0
        prod2 = 1.0
        n_vars = self.number_of_variables()
        
        for i in range(2, n_vars):
            y = x[i] - np.power(x[0], 0.5 * (1.0 + 3.0 * (i - 1.0) / (n_vars - 2.0)))
            y_j = y * y
            if (i + 1) % 2 == 1:  # odd indices (3, 5, 7, ...)
                sum1 += y_j
                prod1 *= np.cos(20.0 * y * np.pi / np.sqrt(i + 1))
            else:  # even indices (4, 6, 8, ...)
                sum2 += y_j
                prod2 *= np.cos(20.0 * y * np.pi / np.sqrt(i + 1))
        
        solution.objectives[0] = x[0] + 2.0 * (4.0 * sum1 - 2.0 * prod1 + 2.0) / np.floor(n_vars / 2.0)
        solution.objectives[1] = 1.0 - np.sqrt(x[0]) + 2.0 * (4.0 * sum2 - 2.0 * prod2 + 2.0) / np.ceil(n_vars / 2.0 - 1.0)
        
        return solution

    def name(self):
        return "UF3"


class UF4(FloatProblem):
    """Problem UF4.

    .. note:: Unconstrained problem. The default number of variables is 30.
    """

    def __init__(self, number_of_variables: int = 30):
        """:param number_of_variables: number of decision variables of the problem."""
        super(UF4, self).__init__()
        self.lower_bound = [0.0] + [-2.0] * (number_of_variables - 1)
        self.upper_bound = [1.0] + [2.0] * (number_of_variables - 1)
        
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f1", "f2"]
    
    def number_of_variables(self) -> int:
        return len(self.lower_bound)
        
    def number_of_objectives(self) -> int:
        return len(self.obj_directions)
        
    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        sum1 = 0.0
        sum2 = 0.0
        count1 = 0
        count2 = 0
        n_vars = self.number_of_variables()
        
        for j in range(2, n_vars + 1):
            yj = x[j - 1] - np.sin(6.0 * np.pi * x[0] + j * np.pi / n_vars)
            hj = abs(yj) / (1.0 + np.exp(2.0 * abs(yj)))
            if j % 2 == 0:  # even indices (2, 4, 6, ...)
                sum2 += hj
                count2 += 1
            else:  # odd indices (3, 5, 7, ...)
                sum1 += hj
                count1 += 1
        
        solution.objectives[0] = x[0] + 2.0 * sum1 / count1
        solution.objectives[1] = 1.0 - x[0] * x[0] + 2.0 * sum2 / count2
        
        return solution

    def name(self):
        return "UF4"


class UF5(FloatProblem):
    """Problem UF5.

    .. note:: Unconstrained problem. The default number of variables is 30.
    """

    def __init__(self, number_of_variables: int = 30, N: int = 10, epsilon: float = 0.1):
        """
        :param number_of_variables: number of decision variables of the problem.
        :param N: controls the number of subcomponents in the problem
        :param epsilon: controls the amplitude of the sine function in the objective
        """
        super(UF5, self).__init__()
        self.lower_bound = [0.0] + [-1.0] * (number_of_variables - 1)
        self.upper_bound = [1.0] * number_of_variables
        self.n = N
        self.epsilon = epsilon
        
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f1", "f2"]
    
    def number_of_variables(self) -> int:
        return len(self.lower_bound)
        
    def number_of_objectives(self) -> int:
        return len(self.obj_directions)
        
    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        sum1 = 0.0
        sum2 = 0.0
        count1 = 0
        count2 = 0
        n_vars = self.number_of_variables()
        
        for j in range(2, n_vars + 1):
            yj = x[j - 1] - np.sin(6.0 * np.pi * x[0] + j * np.pi / n_vars)
            hj = 2.0 * yj * yj - np.cos(4.0 * np.pi * yj) + 1.0
            if j % 2 == 0:  # even indices (2, 4, 6, ...)
                sum2 += hj
                count2 += 1
            else:  # odd indices (3, 5, 7, ...)
                sum1 += hj
                count1 += 1
        
        hj = (0.5 / self.n + self.epsilon) * abs(np.sin(2.0 * self.n * np.pi * x[0]))
        
        solution.objectives[0] = x[0] + hj + 2.0 * sum1 / count1
        solution.objectives[1] = 1.0 - x[0] + hj + 2.0 * sum2 / count2
        
        return solution

    def name(self):
        return "UF5"


class UF6(FloatProblem):
    """Problem UF6.

    .. note:: Unconstrained problem. The default number of variables is 30.
    """

    def __init__(self, number_of_variables: int = 30, N: int = 2, epsilon: float = 0.1):
        """
        :param number_of_variables: number of decision variables of the problem.
        :param N: controls the number of subcomponents in the problem (default: 2)
        :param epsilon: controls the amplitude of the sine function in the objective (default: 0.1)
        """
        super(UF6, self).__init__()
        self.lower_bound = [0.0] + [-1.0] * (number_of_variables - 1)
        self.upper_bound = [1.0] * number_of_variables
        self.n = N
        self.epsilon = epsilon
        
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f1", "f2"]
    
    def number_of_variables(self) -> int:
        return len(self.lower_bound)
        
    def number_of_objectives(self) -> int:
        return len(self.obj_directions)
        
    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        sum1 = 0.0
        sum2 = 0.0
        prod1 = 1.0
        prod2 = 1.0
        count1 = 0
        count2 = 0
        n_vars = self.number_of_variables()
        
        for j in range(2, n_vars + 1):
            yj = x[j - 1] - np.sin(6.0 * np.pi * x[0] + j * np.pi / n_vars)
            pj = np.cos(20.0 * yj * np.pi / np.sqrt(j))
            
            if j % 2 == 0:  # even indices (2, 4, 6, ...)
                sum2 += yj * yj
                prod2 *= pj
                count2 += 1
            else:  # odd indices (3, 5, 7, ...)
                sum1 += yj * yj
                prod1 *= pj
                count1 += 1
        
        hj = 2.0 * (0.5 / self.n + self.epsilon) * np.sin(2.0 * self.n * np.pi * x[0])
        hj = max(0.0, hj)  # Ensure hj is not negative
        
        solution.objectives[0] = x[0] + hj + 2.0 * (4.0 * sum1 - 2.0 * prod1 + 2.0) / count1
        solution.objectives[1] = 1.0 - x[0] + hj + 2.0 * (4.0 * sum2 - 2.0 * prod2 + 2.0) / count2
        
        return solution

    def name(self):
        return "UF6"


class UF7(FloatProblem):
    """Problem UF7.

    .. note:: Unconstrained problem. The default number of variables is 30.
    """

    def __init__(self, number_of_variables: int = 30):
        """:param number_of_variables: number of decision variables of the problem."""
        super(UF7, self).__init__()
        self.lower_bound = [0.0] + [-1.0] * (number_of_variables - 1)
        self.upper_bound = [1.0] * number_of_variables
        
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f1", "f2"]
    
    def number_of_variables(self) -> int:
        return len(self.lower_bound)
        
    def number_of_objectives(self) -> int:
        return len(self.obj_directions)
        
    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        sum1 = 0.0
        sum2 = 0.0
        n_vars = self.number_of_variables()
        
        for i in range(2, n_vars):
            y = x[i] - np.sin(6.0 * np.pi * x[0] + (i + 1) * np.pi / n_vars)
            if (i + 1) % 2 == 1:  # odd indices (3, 5, 7, ...)
                sum1 += y * y
            else:  # even indices (4, 6, 8, ...)
                sum2 += y * y
        
        solution.objectives[0] = np.power(x[0], 0.2) + 2.0 * sum1 / np.floor(n_vars / 2.0)
        solution.objectives[1] = 1.0 - np.power(x[0], 0.2) + 2.0 * sum2 / np.ceil(n_vars / 2.0 - 1.0)
        
        return solution

    def name(self):
        return "UF7"


class UF8(FloatProblem):
    """Problem UF8 - Three-objective problem.

    .. note:: Unconstrained problem. The default number of variables is 30.
    """

    def __init__(self, number_of_variables: int = 30):
        """:param number_of_variables: number of decision variables of the problem."""
        super(UF8, self).__init__()
        self.lower_bound = [0.0, 0.0, 0.0] + [-2.0] * (number_of_variables - 3)
        self.upper_bound = [1.0, 1.0, 1.0] + [2.0] * (number_of_variables - 3)
        
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f1", "f2", "f3"]
    
    def number_of_variables(self) -> int:
        return len(self.lower_bound)
        
    def number_of_objectives(self) -> int:
        return len(self.obj_directions)
        
    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        sum1 = sum2 = sum3 = 0.0
        count1 = count2 = count3 = 0
        n_vars = self.number_of_variables()
        
        for j in range(3, n_vars + 1):  # j = 3 to n_vars (inclusive)
            yj = x[j - 1] - 2.0 * x[1] * np.sin(2.0 * np.pi * x[0] + j * np.pi / n_vars)
            if j % 3 == 1:  # group 1: j=4,7,10,...
                sum1 += yj * yj
                count1 += 1
            elif j % 3 == 2:  # group 2: j=5,8,11,...
                sum2 += yj * yj
                count2 += 1
            else:  # group 3: j=3,6,9,...
                sum3 += yj * yj
                count3 += 1
        
        # Avoid division by zero (though unlikely with n_vars >= 3)
        count1 = max(1, count1)
        count2 = max(1, count2)
        count3 = max(1, count3)
        
        solution.objectives[0] = np.cos(0.5 * np.pi * x[0]) * np.cos(0.5 * np.pi * x[1]) + 2.0 * sum1 / count1
        solution.objectives[1] = np.cos(0.5 * np.pi * x[0]) * np.sin(0.5 * np.pi * x[1]) + 2.0 * sum2 / count2
        solution.objectives[2] = np.sin(0.5 * np.pi * x[0]) + 2.0 * sum3 / count3
        
        return solution

    def name(self):
        return "UF8"


class UF9(FloatProblem):
    """Problem UF9 - Three-objective problem with variable bounds.

    .. note:: Unconstrained problem. The default number of variables is 30.
    """

    def __init__(self, number_of_variables: int = 30, epsilon: float = 0.1):
        """
        :param number_of_variables: number of decision variables of the problem.
        :param epsilon: controls the shape of the Pareto front (default: 0.1)
        """
        super(UF9, self).__init__()
        self.lower_bound = [0.0, 0.0, 0.0] + [-2.0] * (number_of_variables - 3)
        self.upper_bound = [1.0, 1.0, 1.0] + [2.0] * (number_of_variables - 3)
        self.epsilon = epsilon
        
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f1", "f2", "f3"]
    
    def number_of_variables(self) -> int:
        return len(self.lower_bound)
        
    def number_of_objectives(self) -> int:
        return len(self.obj_directions)
        
    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        sum1 = sum2 = sum3 = 0.0
        count1 = count2 = count3 = 0
        n_vars = self.number_of_variables()
        
        for j in range(3, n_vars + 1):  # j = 3 to n_vars (inclusive)
            yj = x[j - 1] - 2.0 * x[1] * np.sin(2.0 * np.pi * x[0] + j * np.pi / n_vars)
            hj = 2.0 * yj * yj - np.cos(4.0 * np.pi * yj) + 1.0
            if j % 3 == 1:  # group 1: j=4,7,10,...
                sum1 += hj
                count1 += 1
            elif j % 3 == 2:  # group 2: j=5,8,11,...
                sum2 += hj
                count2 += 1
            else:  # group 3: j=3,6,9,...
                sum3 += hj
                count3 += 1
        
        # Avoid division by zero
        count1 = max(1, count1)
        count2 = max(1, count2)
        count3 = max(1, count3)
        
        yj = (1.0 + self.epsilon) * (1.0 - 4.0 * (2.0 * x[0] - 1.0) ** 2)
        yj = max(0.0, yj)
        
        solution.objectives[0] = 0.5 * (yj + 2.0 * x[0]) * x[1] + 2.0 * sum1 / count1
        solution.objectives[1] = 0.5 * (yj - 2.0 * x[0] + 2.0) * x[1] + 2.0 * sum2 / count2
        solution.objectives[2] = 1.0 - x[1] + 2.0 * sum3 / count3
        
        return solution

    def name(self):
        return "UF9"


class UF10(FloatProblem):
    """Problem UF10 - Three-objective problem with complex interactions.

    .. note:: Unconstrained problem. The default number of variables is 30.
    """

    def __init__(self, number_of_variables: int = 30):
        """:param number_of_variables: number of decision variables of the problem."""
        super(UF10, self).__init__()
        self.lower_bound = [0.0, 0.0, 0.0] + [-2.0] * (number_of_variables - 3)
        self.upper_bound = [1.0, 1.0, 1.0] + [2.0] * (number_of_variables - 3)
        
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f1", "f2", "f3"]
    
    def number_of_variables(self) -> int:
        return len(self.lower_bound)
        
    def number_of_objectives(self) -> int:
        return len(self.obj_directions)
        
    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        sum1 = sum2 = sum3 = 0.0
        count1 = count2 = count3 = 0
        n_vars = self.number_of_variables()
        
        for j in range(3, n_vars + 1):  # j = 3 to n_vars (inclusive)
            yj = x[j - 1] - 2.0 * x[1] * np.sin(2.0 * np.pi * x[0] + j * np.pi / n_vars)
            hj = 4.0 * yj * yj - np.cos(8.0 * np.pi * yj) + 1.0
            if j % 3 == 1:  # group 1: j=4,7,10,...
                sum1 += hj
                count1 += 1
            elif j % 3 == 2:  # group 2: j=5,8,11,...
                sum2 += hj
                count2 += 1
            else:  # group 3: j=3,6,9,...
                sum3 += hj
                count3 += 1
        
        # Avoid division by zero
        count1 = max(1, count1)
        count2 = max(1, count2)
        count3 = max(1, count3)
        
        solution.objectives[0] = np.cos(0.5 * np.pi * x[0]) * np.cos(0.5 * np.pi * x[1]) + 2.0 * sum1 / count1
        solution.objectives[1] = np.cos(0.5 * np.pi * x[0]) * np.sin(0.5 * np.pi * x[1]) + 2.0 * sum2 / count2
        solution.objectives[2] = np.sin(0.5 * np.pi * x[0]) + 2.0 * sum3 / count3
        
        return solution

    def name(self):
        return "UF10"
