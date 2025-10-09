import math
from math import sqrt
from typing import Sequence

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


def get_closest_value(target_array: Sequence[float], comp_value: float) -> float:
    """Return the value in target_array that is closest to comp_value.

    This is a direct translation of the provided Java method. It assumes
    target_array contains at least one element; otherwise, raises ValueError.

    Args:
        target_array: A non-empty sequence of floats.
        comp_value: The value to compare against.

    Returns:
        The element of target_array with minimum absolute difference to comp_value.
    """
    if not target_array:
        raise ValueError("target_array must be a non-empty sequence")

    closest_value = target_array[0]
    min_diff_value = abs(target_array[0] - comp_value)

    for i in range(1, len(target_array)):
        tmp_diff_value = abs(target_array[i] - comp_value)
        if tmp_diff_value < min_diff_value:
            min_diff_value = tmp_diff_value
            closest_value = target_array[i]

    return float(closest_value)


class RE21(FloatProblem):
    """Problem RE21 from:
    Ryoji Tanabe and Hisao Ishibuchi, "An easy-to-use real-world multi-objective optimization
    problem suite", Applied Soft Computing, Vol. 89, 106078 (2020).
    DOI: https://doi.org/10.1016/j.asoc.2020.106078

    This is a two-objective, unconstrained, continuous problem with 4 decision variables.
    """

    def __init__(self):
        super(RE21, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(x)", "f(y)"]

        number_of_variables = 4

        f = 10.0
        sigma = 10.0
        tmp_var = f / sigma

        upper = [3.0 * tmp_var for _ in range(number_of_variables)]
        lower = [0.0 for _ in range(number_of_variables)]
        lower[0] = tmp_var
        lower[1] = sqrt(2.0) * tmp_var
        lower[2] = sqrt(2.0) * tmp_var
        lower[3] = tmp_var

        self.lower_bound = lower
        self.upper_bound = upper

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = solution.variables[0]
        x2 = solution.variables[1]
        x3 = solution.variables[2]
        x4 = solution.variables[3]

        f = 10.0
        e = 200000.0
        l = 200.0

        solution.objectives[0] = l * ((2.0 * x1) + sqrt(2.0) * x2 + sqrt(x3) + x4)
        solution.objectives[1] = ((f * l) / e) * (
            (2.0 / x1)
            + (2.0 * sqrt(2.0) / x2)
            - (2.0 * sqrt(2.0) / x3)
            + (2.0 / x4)
        )

        return solution

    def name(self) -> str:
        return "RE21"


class RE22(FloatProblem):
    """Problem RE22 from:
    Ryoji Tanabe and Hisao Ishibuchi, "An easy-to-use real-world multi-objective optimization
    problem suite", Applied Soft Computing, Vol. 89, 106078 (2020).
    DOI: https://doi.org/10.1016/j.asoc.2020.106078

    This is a two-objective, unconstrained, mixed-integer problem with 3 decision variables.
    """

    AS_FEASIBLE_INTEGERS = [
        0.20, 0.31, 0.40, 0.44, 0.60, 0.62, 0.79, 0.80, 0.88, 0.93, 1.0, 1.20, 1.24, 1.32, 1.40, 1.55,
        1.58, 1.60, 1.76, 1.80, 1.86, 2.0, 2.17, 2.20, 2.37, 2.40, 2.48, 2.60, 2.64, 2.79, 2.80, 3.0,
        3.08, 3.10, 3.16, 3.41, 3.52, 3.60, 3.72, 3.95, 3.96, 4.0, 4.03, 4.20, 4.34, 4.40, 4.65, 4.74,
        4.80, 4.84, 5.0, 5.28, 5.40, 5.53, 5.72, 6.0, 6.16, 6.32, 6.60, 7.11, 7.20, 7.80, 7.90, 8.0,
        8.40, 8.69, 9.0, 9.48, 10.27, 11.0, 11.06, 11.85, 12.0, 13.0, 14.0, 15.0
    ]

    def __init__(self):
        super(RE22, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(x)", "f(y)"]

        self.lower_bound = [0.2, 0.0, 0.0]
        self.upper_bound = [15.0, 20.0, 40.0]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = get_closest_value(self.AS_FEASIBLE_INTEGERS, solution.variables[0])
        x2 = solution.variables[1]
        x3 = solution.variables[2]

        g = [0.0, 0.0]
        g[0] = (x1 * x3) - 7.735 * ((x1 * x1) / x2) - 180.0
        g[1] = 4.0 - (x3 / x2)

        for i in range(len(g)):
            if g[i] < 0.0:
                g[i] = -g[i]
            else:
                g[i] = 0.0

        solution.objectives[0] = (29.4 * x1) + (0.6 * x2 * x3)
        solution.objectives[1] = g[0] + g[1]

        return solution

    def name(self) -> str:
        return "RE22"


class RE23(FloatProblem):
    """Problem RE23 from:
    Ryoji Tanabe and Hisao Ishibuchi, "An easy-to-use real-world multi-objective optimization
    problem suite", Applied Soft Computing, Vol. 89, 106078 (2020).
    DOI: https://doi.org/10.1016/j.asoc.2020.106078

    This is a two-objective, unconstrained, mixed-integer problem with 4 decision variables.
    """

    def __init__(self):
        super(RE23, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(x)", "f(y)"]

        self.lower_bound = [1.0, 1.0, 10.0, 10.0]
        self.upper_bound = [100.0, 100.0, 200.0, 240.0]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = 0.0625 * round(solution.variables[0] / 0.0625)  # Round to nearest 0.0625
        x2 = 0.0625 * round(solution.variables[1] / 0.0625)  # Round to nearest 0.0625
        x3 = solution.variables[2]
        x4 = solution.variables[3]

        g = [0.0, 0.0, 0.0]
        g[0] = x1 - (0.0193 * x3)
        g[1] = x2 - (0.00954 * x3)
        g[2] = (math.pi * x3 * x3 * x4) + ((4.0 / 3.0) * (math.pi * x3 * x3 * x3)) - 1296000.0

        for i in range(len(g)):
            if g[i] < 0.0:
                g[i] = -g[i]
            else:
                g[i] = 0.0

        solution.objectives[0] = (
            (0.6224 * x1 * x3 * x4)
            + (1.7781 * x2 * x3 * x3)
            + (3.1661 * x1 * x1 * x4)
            + (19.84 * x1 * x1 * x3)
        )
        solution.objectives[1] = g[0] + g[1] + g[2]

        return solution

    def name(self) -> str:
        return "RE23"


class RE24(FloatProblem):
    """Problem RE24 from:
    Ryoji Tanabe and Hisao Ishibuchi, "An easy-to-use real-world multi-objective optimization
    problem suite", Applied Soft Computing, Vol. 89, 106078 (2020).
    DOI: https://doi.org/10.1016/j.asoc.2020.106078

    This is a two-objective, unconstrained, continuous problem with 2 decision variables.
    """

    def __init__(self):
        super(RE24, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(x)", "f(y)"]

        self.lower_bound = [0.5, 0.5]
        self.upper_bound = [4.0, 50.0]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = solution.variables[0]
        x2 = solution.variables[1]

        e = 700000.0
        sigma_b_max = 700.0
        tau_max = 450.0
        delta_max = 1.5
        sigma_k = (e * x1 * x1) / 100.0
        sigma_b = 4500.0 / (x1 * x2)
        tau = 1800.0 / x2
        delta = (56.2 * 10000.0) / (e * x1 * x2 * x2)

        g = [0.0, 0.0, 0.0, 0.0]
        g[0] = 1.0 - (sigma_b / sigma_b_max)
        g[1] = 1.0 - (tau / tau_max)
        g[2] = 1.0 - (delta / delta_max)
        g[3] = 1.0 - (sigma_b / sigma_k)

        for i in range(len(g)):
            if g[i] < 0.0:
                g[i] = -g[i]
            else:
                g[i] = 0.0

        solution.objectives[0] = x1 + (120.0 * x2)
        solution.objectives[1] = g[0] + g[1] + g[2] + g[3]

        return solution

    def name(self) -> str:
        return "RE24"


class RE25(FloatProblem):
    """Problem RE25 from:
    Ryoji Tanabe and Hisao Ishibuchi, "An easy-to-use real-world multi-objective optimization
    problem suite", Applied Soft Computing, Vol. 89, 106078 (2020).
    DOI: https://doi.org/10.1016/j.asoc.2020.106078

    This is a two-objective, unconstrained, mixed-integer problem with 3 decision variables.
    """

    # Feasible values for the diameter (x3)
    DIAMETER_FEASIBLE_VALUES = [
        0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014, 0.015, 0.0162, 0.0173, 0.018, 0.02, 0.023,
        0.025, 0.028, 0.032, 0.035, 0.041, 0.047, 0.054, 0.063, 0.072, 0.08, 0.092, 0.105, 0.12, 0.135,
        0.148, 0.162, 0.177, 0.192, 0.207, 0.225, 0.244, 0.263, 0.283, 0.307, 0.331, 0.362, 0.394,
        0.4375, 0.5
    ]

    def __init__(self):
        super(RE25, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(x)", "f(y)"]

        self.lower_bound = [1.0, 0.6, 0.09]
        self.upper_bound = [70.0, 3.0, 0.5]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = round(solution.variables[0])  # Integer value
        x2 = solution.variables[1]
        x3 = get_closest_value(self.DIAMETER_FEASIBLE_VALUES, solution.variables[2])

        g = [0.0] * 6
        
        cf = ((4.0 * (x2 / x3) - 1) / (4.0 * (x2 / x3) - 4)) + (0.615 * x3 / x2)
        f_max = 1000.0
        S = 189000.0
        G = 11.5e6
        K = (G * x3**4) / (8 * x1 * x2**3)
        l_max = 14.0
        lf = (f_max / K) + 1.05 * (x1 + 2) * x3
        Fp = 300.0
        sigma_p = Fp / K
        sigma_pm = 6.0
        sigma_w = 1.25

        g[0] = -((8 * cf * f_max * x2) / (math.pi * x3**3)) + S
        g[1] = -lf + l_max
        g[2] = -3 + (x2 / x3)
        g[3] = -sigma_p + sigma_pm
        g[4] = -sigma_p - ((f_max - Fp) / K) - 1.05 * (x1 + 2) * x3 + lf
        g[5] = sigma_w - ((f_max - Fp) / K)

        for i in range(len(g)):
            if g[i] < 0.0:
                g[i] = -g[i]
            else:
                g[i] = 0.0

        solution.objectives[0] = (math.pi**2 * x2 * x3**2 * (x1 + 2)) / 4.0
        solution.objectives[1] = sum(g)

        return solution

    def name(self) -> str:
        return "RE25"


class RE31(FloatProblem):
    """Problem RE31 from:
    Ryoji Tanabe and Hisao Ishibuchi, "An easy-to-use real-world multi-objective optimization
    problem suite", Applied Soft Computing, Vol. 89, 106078 (2020).
    DOI: https://doi.org/10.1016/j.asoc.2020.106078

    This is a three-objective, unconstrained, continuous problem with 3 decision variables.
    """

    def __init__(self):
        super(RE31, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(x)", "f(y)", "f(z)"]

        self.lower_bound = [0.00001, 0.00001, 1.0]
        self.upper_bound = [100.0, 100.0, 3.0]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = solution.variables[0]
        x2 = solution.variables[1]
        x3 = solution.variables[2]

        # Calculate first two objectives
        solution.objectives[0] = x1 * math.sqrt(16.0 + (x3 * x3)) + x2 * math.sqrt(1.0 + x3 * x3)
        solution.objectives[1] = (20.0 * math.sqrt(16.0 + (x3 * x3))) / (x1 * x3)

        # Calculate constraint violations
        g = [0.0, 0.0, 0.0]
        g[0] = 0.1 - solution.objectives[0]
        g[1] = 100000.0 - solution.objectives[1]
        g[2] = 100000.0 - ((80.0 * math.sqrt(1.0 + x3 * x3)) / (x3 * x2))

        # Convert constraint violations to penalties
        for i in range(len(g)):
            if g[i] < 0.0:
                g[i] = -g[i]
            else:
                g[i] = 0.0

        # Third objective is the sum of constraint violations
        solution.objectives[2] = sum(g)

        return solution

    def name(self) -> str:
        return "RE31"


class RE32(FloatProblem):
    """Problem RE32 from:
    Ryoji Tanabe and Hisao Ishibuchi, "An easy-to-use real-world multi-objective optimization
    problem suite", Applied Soft Computing, Vol. 89, 106078 (2020).
    DOI: https://doi.org/10.1016/j.asoc.2020.106078

    This is a three-objective, unconstrained, continuous problem with 4 decision variables.
    """

    def __init__(self):
        super(RE32, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(x)", "f(y)", "f(z)"]

        self.lower_bound = [0.125, 0.1, 0.1, 0.125]
        self.upper_bound = [5.0, 10.0, 10.0, 5.0]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = solution.variables[0]
        x2 = solution.variables[1]
        x3 = solution.variables[2]
        x4 = solution.variables[3]

        P = 6000.0
        L = 14.0
        E = 30.0e6
        G = 12.0e6
        tau_max = 13600.0
        sigma_max = 30000.0

        # Calculate first two objectives
        solution.objectives[0] = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
        solution.objectives[1] = (4 * P * L**3) / (E * x4 * x3**3)

        # Calculate constraint violations
        M = P * (L + (x2 / 2.0))
        tmp_var = ((x2 * x2) / 4.0) + ((x1 + x3) / 2.0)**2
        R = math.sqrt(tmp_var)
        tmp_var = ((x2 * x2) / 12.0) + ((x1 + x3) / 2.0)**2
        J = 2 * math.sqrt(2) * x1 * x2 * tmp_var

        tau_dash_dash = (M * R) / J
        tau_dash = P / (math.sqrt(2) * x1 * x2)
        tmp_var = (tau_dash**2 + 
                  ((2 * tau_dash * tau_dash_dash * x2) / (2 * R)) + 
                  (tau_dash_dash**2))
        tau = math.sqrt(tmp_var)
        sigma = (6 * P * L) / (x4 * x3 * x3)
        tmp_var = 4.013 * E * math.sqrt((x3**2 * x4**6) / 36.0) / (L * L)
        tmp_var2 = (x3 / (2 * L)) * math.sqrt(E / (4 * G))
        PC = tmp_var * (1 - tmp_var2)

        g = [0.0, 0.0, 0.0, 0.0]
        g[0] = tau_max - tau
        g[1] = sigma_max - sigma
        g[2] = x4 - x1
        g[3] = PC - P

        # Convert constraint violations to penalties
        for i in range(len(g)):
            if g[i] < 0.0:
                g[i] = -g[i]
            else:
                g[i] = 0.0

        # Third objective is the sum of constraint violations
        solution.objectives[2] = sum(g)

        return solution

    def name(self) -> str:
        return "RE32"


class RE33(FloatProblem):
    """Problem RE33 from:
    Ryoji Tanabe and Hisao Ishibuchi, "An easy-to-use real-world multi-objective optimization
    problem suite", Applied Soft Computing, Vol. 89, 106078 (2020).
    DOI: https://doi.org/10.1016/j.asoc.2020.106078

    This is a three-objective, unconstrained, continuous problem with 4 decision variables.
    """

    def __init__(self):
        super(RE33, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(x)", "f(y)", "f(z)"]

        self.lower_bound = [55.0, 75.0, 1000.0, 11.0]
        self.upper_bound = [80.0, 110.0, 3000.0, 20.0]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = solution.variables[0]
        x2 = solution.variables[1]
        x3 = solution.variables[2]
        x4 = solution.variables[3]

        # Calculate first two objectives
        solution.objectives[0] = 4.9e-5 * (x2**2 - x1**2) * (x4 - 1.0)
        solution.objectives[1] = (9.82e6 * (x2**2 - x1**2)) / (x3 * x4 * (x2**3 - x1**3))

        # Calculate constraint violations
        g = [0.0, 0.0, 0.0, 0.0]
        g[0] = (x2 - x1) - 20.0
        g[1] = 0.4 - (x3 / (math.pi * (x2**2 - x1**2)))
        g[2] = 1.0 - (2.22e-3 * x3 * (x2**3 - x1**3)) / ((x2**2 - x1**2) ** 2)
        g[3] = (2.66e-2 * x3 * x4 * (x2**3 - x1**3)) / (x2**2 - x1**2) - 900.0

        # Convert constraint violations to penalties
        for i in range(len(g)):
            if g[i] < 0.0:
                g[i] = -g[i]
            else:
                g[i] = 0.0

        # Third objective is the sum of constraint violations
        solution.objectives[2] = sum(g)

        return solution

    def name(self) -> str:
        return "RE33"


class RE34(FloatProblem):
    """Problem RE34 from:
    Ryoji Tanabe and Hisao Ishibuchi, "An easy-to-use real-world multi-objective optimization
    problem suite", Applied Soft Computing, Vol. 89, 106078 (2020).
    DOI: https://doi.org/10.1016/j.asoc.2020.106078

    This is a three-objective, unconstrained, continuous problem with 5 decision variables.
    """

    def __init__(self, number_of_variables: int = 5):
        super(RE34, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(x)", "f(y)", "f(z)"]

        # All variables have the same bounds [1.0, 3.0]
        self.lower_bound = [1.0] * number_of_variables
        self.upper_bound = [3.0] * number_of_variables

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = solution.variables[0]
        x2 = solution.variables[1]
        x3 = solution.variables[2]
        x4 = solution.variables[3]
        x5 = solution.variables[4]

        # First objective
        solution.objectives[0] = (
            1640.2823
            + (2.3573285 * x1)
            + (2.3220035 * x2)
            + (4.5688768 * x3)
            + (7.7213633 * x4)
            + (4.4559504 * x5)
        )

        # Second objective
        solution.objectives[1] = (
            6.5856
            + (1.15 * x1)
            - (1.0427 * x2)
            + (0.9738 * x3)
            + (0.8364 * x4)
            - (0.3695 * x1 * x4)
            + (0.0861 * x1 * x5)
            + (0.3628 * x2 * x4)
            - (0.1106 * x1 * x1)
            - (0.3437 * x3 * x3)
            + (0.1764 * x4 * x4)
        )

        # Third objective
        solution.objectives[2] = (
            -0.0551
            + (0.0181 * x1)
            + (0.1024 * x2)
            + (0.0421 * x3)
            - (0.0073 * x1 * x2)
            + (0.024 * x2 * x3)
            - (0.0118 * x2 * x4)
            - (0.0204 * x3 * x4)
            - (0.008 * x3 * x5)
            - (0.0241 * x2 * x2)
            + (0.0109 * x4 * x4)
        )

        return solution

    def name(self) -> str:
        return "RE34"


class RE35(FloatProblem):
    """Problem RE35 from:
    Ryoji Tanabe and Hisao Ishibuchi, "An easy-to-use real-world multi-objective optimization
    problem suite", Applied Soft Computing, Vol. 89, 106078 (2020).
    DOI: https://doi.org/10.1016/j.asoc.2020.106078

    This is a three-objective, unconstrained, mixed-integer problem with 7 decision variables.
    """

    def __init__(self):
        super(RE35, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(x)", "f(y)", "f(z)"]

        self.lower_bound = [2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0]
        self.upper_bound = [3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = solution.variables[0]
        x2 = solution.variables[1]
        x3 = round(solution.variables[2])  # Integer value (using round as in Java's Math.rint)
        x4 = solution.variables[3]
        x5 = solution.variables[4]
        x6 = solution.variables[5]
        x7 = solution.variables[6]

        # First objective
        solution.objectives[0] = (
            0.7854 * x1 * (x2**2) * (((10.0 * x3**2) / 3.0) + (14.933 * x3) - 43.0934)
            - 1.508 * x1 * (x6**2 + x7**2)
            + 7.477 * (x6**3 + x7**3)
            + 0.7854 * (x4 * x6**2 + x5 * x7**2)
        )

        # Second objective
        tmp_var = ((745.0 * x4) / (x2 * x3))**2 + 1.69e7
        solution.objectives[1] = math.sqrt(tmp_var) / (0.1 * x6**3)

        # Calculate constraint violations
        g = [0.0] * 11
        g[0] = -(1.0 / (x1 * x2**2 * x3)) + 1.0 / 27.0
        g[1] = -(1.0 / (x1 * x2**2 * x3**2)) + 1.0 / 397.5
        g[2] = -(x4**3) / (x2 * x3 * x6**4) + 1.0 / 1.93
        g[3] = -(x5**3) / (x2 * x3 * x7**4) + 1.0 / 1.93
        g[4] = -(x2 * x3) + 40.0
        g[5] = -(x1 / x2) + 12.0
        g[6] = -5.0 + (x1 / x2)
        g[7] = -1.9 + x4 - 1.5 * x6
        g[8] = -1.9 + x5 - 1.1 * x7
        g[9] = -solution.objectives[1] + 1300.0
        
        tmp_var = ((745.0 * x5) / (x2 * x3))**2 + 1.575e8
        g[10] = -math.sqrt(tmp_var) / (0.1 * x7**3) + 1100.0

        # Convert constraint violations to penalties
        for i in range(len(g)):
            if g[i] < 0.0:
                g[i] = -g[i]
            else:
                g[i] = 0.0

        # Third objective is the sum of constraint violations
        solution.objectives[2] = sum(g)

        return solution

    def name(self) -> str:
        return "RE35"


class RE36(FloatProblem):
    """Problem RE36 from:
    Ryoji Tanabe and Hisao Ishibuchi, "An easy-to-use real-world multi-objective optimization
    problem suite", Applied Soft Computing, Vol. 89, 106078 (2020).
    DOI: https://doi.org/10.1016/j.asoc.2020.106078

    This is a three-objective, unconstrained, discrete problem with 4 decision variables.
    """

    def __init__(self):
        super(RE36, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(x)", "f(y)", "f(z)"]

        self.lower_bound = [12.0, 12.0, 12.0, 12.0]
        self.upper_bound = [60.0, 60.0, 60.0, 60.0]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = round(solution.variables[0])
        x2 = round(solution.variables[1])
        x3 = round(solution.variables[2])
        x4 = round(solution.variables[3])

        # First objective
        solution.objectives[0] = abs(6.931 - ((x3 / x1) * (x4 / x2)))

        # Second objective (maximum of the four variables)
        solution.objectives[1] = max(x1, x2, x3, x4)

        # Third objective (constraint violation)
        g = 0.5 - (solution.objectives[0] / 6.931)
        solution.objectives[2] = -g if g < 0.0 else 0.0

        return solution

    def name(self) -> str:
        return "RE36"


class RE37(FloatProblem):
    """Problem RE37 from:
    Ryoji Tanabe and Hisao Ishibuchi, "An easy-to-use real-world multi-objective optimization
    problem suite", Applied Soft Computing, Vol. 89, 106078 (2020).
    DOI: https://doi.org/10.1016/j.asoc.2020.106078

    This is a three-objective, unconstrained, continuous problem with 4 decision variables.
    """

    def __init__(self):
        super(RE37, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(x)", "f(y)", "f(z)"]

        number_of_variables = 4
        self.lower_bound = [0.0] * number_of_variables
        self.upper_bound = [1.0] * number_of_variables

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x_alpha = solution.variables[0]
        x_ha = solution.variables[1]
        x_oa = solution.variables[2]
        x_optt = solution.variables[3]

        # First objective
        solution.objectives[0] = (
            0.692
            + (0.477 * x_alpha)
            - (0.687 * x_ha)
            - (0.080 * x_oa)
            - (0.0650 * x_optt)
            - (0.167 * x_alpha * x_alpha)
            - (0.0129 * x_ha * x_alpha)
            + (0.0796 * x_ha * x_ha)
            - (0.0634 * x_oa * x_alpha)
            - (0.0257 * x_oa * x_ha)
            + (0.0877 * x_oa * x_oa)
            - (0.0521 * x_optt * x_alpha)
            + (0.00156 * x_optt * x_ha)
            + (0.00198 * x_optt * x_oa)
            + (0.0184 * x_optt * x_optt)
        )

        # Second objective
        solution.objectives[1] = (
            0.153
            - (0.322 * x_alpha)
            + (0.396 * x_ha)
            + (0.424 * x_oa)
            + (0.0226 * x_optt)
            + (0.175 * x_alpha * x_alpha)
            + (0.0185 * x_ha * x_alpha)
            - (0.0701 * x_ha * x_ha)
            - (0.251 * x_oa * x_alpha)
            + (0.179 * x_oa * x_ha)
            + (0.0150 * x_oa * x_oa)
            + (0.0134 * x_optt * x_alpha)
            + (0.0296 * x_optt * x_ha)
            + (0.0752 * x_optt * x_oa)
            + (0.0192 * x_optt * x_optt)
        )

        # Third objective
        solution.objectives[2] = (
            0.370
            - (0.205 * x_alpha)
            + (0.0307 * x_ha)
            + (0.108 * x_oa)
            + (1.019 * x_optt)
            - (0.135 * x_alpha * x_alpha)
            + (0.0141 * x_ha * x_alpha)
            + (0.0998 * x_ha * x_ha)
            + (0.208 * x_oa * x_alpha)
            - (0.0301 * x_oa * x_ha)
            - (0.226 * x_oa * x_oa)
            + (0.353 * x_optt * x_alpha)
            - (0.0497 * x_optt * x_oa)
            - (0.423 * x_optt * x_optt)
            + (0.202 * x_ha * x_alpha * x_alpha)
            - (0.281 * x_oa * x_alpha * x_alpha)
            - (0.342 * x_ha * x_ha * x_alpha)
            - (0.245 * x_ha * x_ha * x_oa)
            + (0.281 * x_oa * x_oa * x_ha)
            - (0.184 * x_optt * x_optt * x_alpha)
            - (0.281 * x_ha * x_alpha * x_oa)
        )

        return solution

    def name(self) -> str:
        return "RE37"


class RE41(FloatProblem):
    """Problem RE41 from:
    Ryoji Tanabe and Hisao Ishibuchi, "An easy-to-use real-world multi-objective optimization
    problem suite", Applied Soft Computing, Vol. 89, 106078 (2020).
    DOI: https://doi.org/10.1016/j.asoc.2020.106078

    This is a four-objective, unconstrained, discrete problem with 7 decision variables and 10 constraints.
    """

    def __init__(self):
        super(RE41, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(w)", "f(x)", "f(y)", "f(z)"]
        self.number_of_original_constraints = 10

        self.lower_bound = [0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4]
        self.upper_bound = [1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = round(solution.variables[0])  # x_alpha
        x2 = round(solution.variables[1])  # x_HA
        x3 = round(solution.variables[2])  # x_OA
        x4 = round(solution.variables[3])  # x_OPTT
        x5 = round(solution.variables[4])  # x_alpha2
        x6 = round(solution.variables[5])  # x_HA2
        x7 = round(solution.variables[6])  # x_OA2

        # First objective (cost)
        solution.objectives[0] = (
            1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 0.00001 * x6 + 2.73 * x7
        )

        # Second objective (reliability)
        solution.objectives[1] = 4.72 - 0.5 * x4 - 0.19 * x2 * x3

        # Third objective (average of Vmbp and Vfd)
        Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
        solution.objectives[2] = 0.5 * (Vmbp + Vfd)

        # Constraint handling for the fourth objective
        g = [0.0] * self.number_of_original_constraints
        
        g[0] = 1 - (1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3)
        g[1] = 0.32 - (0.261 - 0.0159 * x1 * x2 - 0.06486 * x1 - 0.019 * x2 * x7 + 
                       0.0144 * x3 * x5 + 0.0154464 * x6)
        g[2] = 0.32 - (0.214 + 0.00817 * x5 - 0.045195 * x1 - 0.0135168 * x1 + 
                        0.03099 * x2 * x6 - 0.018 * x2 * x7 + 0.007176 * x3 + 
                        0.023232 * x3 - 0.00364 * x5 * x6 - 0.018 * x2 * x2)
        g[3] = 0.32 - (0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2)
        g[4] = 32 - (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
        g[5] = 32 - (33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728)
        g[6] = 32 - (46.36 - 9.9 * x2 - 4.4505 * x1)
        g[7] = 4 - solution.objectives[1]  # Constraint on second objective
        g[8] = 9.9 - Vmbp
        g[9] = 15.7 - Vfd

        # Convert constraints to violations (only count when g[i] > 0)
        for i in range(self.number_of_original_constraints):
            g[i] = max(0.0, g[i])

        # Fourth objective (sum of constraint violations)
        solution.objectives[3] = sum(g)

        return solution

    def name(self) -> str:
        return "RE41"


class RE42(FloatProblem):
    """Problem RE42 from:
    Ryoji Tanabe and Hisao Ishibuchi, "An easy-to-use real-world multi-objective optimization
    problem suite", Applied Soft Computing, Vol. 89, 106078 (2020).
    DOI: https://doi.org/10.1016/j.asoc.2020.106078

    This is a four-objective, unconstrained, continuous problem with 6 decision variables and 9 constraints.
    The problem represents a ship design optimization problem.
    """

    def __init__(self):
        super(RE42, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(w)", "f(x)", "f(y)", "f(z)"]
        self.number_of_original_constraints = 9

        self.lower_bound = [150.0, 20.0, 13.0, 10.0, 14.0, 0.63]
        self.upper_bound = [274.32, 32.31, 25.0, 11.71, 18.0, 0.75]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x_L = solution.variables[0]  # Length
        x_B = solution.variables[1]  # Beam
        x_D = solution.variables[2]  # Depth
        x_T = solution.variables[3]  # Draft
        x_Vk = solution.variables[4]  # Speed
        x_CB = solution.variables[5]  # Block coefficient

        # Intermediate calculations
        displacement = 1.025 * x_L * x_B * x_T * x_CB
        V = 0.5144 * x_Vk
        g = 9.8065
        Fn = V / (g * x_L) ** 0.5
        a = 4977.06 * x_CB**2 - 8105.61 * x_CB + 4456.51
        b = -10847.2 * x_CB**2 + 12817.0 * x_CB - 6960.32

        # Power calculation
        power = (displacement ** (2.0/3.0) * x_Vk**3) / (a + (b * Fn))
        
        # Weight calculations
        outfit_weight = 1.0 * (x_L**0.8) * (x_B**0.6) * (x_D**0.3) * (x_CB**0.1)
        steel_weight = 0.034 * (x_L**1.7) * (x_B**0.7) * (x_D**0.4) * (x_CB**0.5)
        machinery_weight = 0.17 * (power**0.9)
        light_ship_weight = steel_weight + outfit_weight + machinery_weight

        # Cost calculations
        ship_cost = 1.3 * ((2000.0 * steel_weight**0.85) + (3500.0 * outfit_weight) + (2400.0 * power**0.8))
        capital_costs = 0.2 * ship_cost

        # Deadweight tonnage
        DWT = displacement - light_ship_weight

        # Running costs
        running_costs = 40000.0 * (DWT**0.3)

        # Voyage calculations
        round_trip_miles = 5000.0
        sea_days = (round_trip_miles / 24.0) * x_Vk
        handling_rate = 8000.0

        daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
        fuel_price = 100.0
        fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
        port_cost = 6.3 * (DWT**0.8)

        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous_DWT = 2.0 * (DWT**0.5)

        cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
        port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
        RTPA = 350.0 / (sea_days + port_days)

        voyage_costs = (fuel_cost + port_cost) * RTPA
        annual_costs = capital_costs + running_costs + voyage_costs
        annual_cargo = cargo_DWT * RTPA

        # Objectives
        solution.objectives[0] = annual_costs / annual_cargo  # Cost per ton-mile
        solution.objectives[1] = light_ship_weight  # Light ship weight
        solution.objectives[2] = -annual_cargo  # Negative annual cargo (to be minimized)

        # Constraint handling for the fourth objective
        constraint_funcs = [0.0] * self.number_of_original_constraints
        
        constraint_funcs[0] = (x_L / x_B) - 6.0
        constraint_funcs[1] = -(x_L / x_D) + 15.0
        constraint_funcs[2] = -(x_L / x_T) + 19.0
        constraint_funcs[3] = 0.45 * (DWT**0.31) - x_T
        constraint_funcs[4] = 0.7 * x_D + 0.7 - x_T
        constraint_funcs[5] = 500000.0 - DWT
        constraint_funcs[6] = DWT - 3000.0
        constraint_funcs[7] = 0.32 - Fn

        # Stability constraint
        KB = 0.53 * x_T
        BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB)
        KG = 1.0 + 0.52 * x_D
        constraint_funcs[8] = (KB + BMT - KG) - (0.07 * x_B)

        # Convert constraints to violations (only count when constraint_funcs[i] > 0)
        for i in range(self.number_of_original_constraints):
            constraint_funcs[i] = max(0.0, constraint_funcs[i])

        # Fourth objective (sum of constraint violations)
        solution.objectives[3] = sum(constraint_funcs)

        return solution

    def name(self) -> str:
        return "RE42"


class RE61(FloatProblem):
    """Problem RE61 from:
    Ryoji Tanabe and Hisao Ishibuchi, "An easy-to-use real-world multi-objective optimization
    problem suite", Applied Soft Computing, Vol. 89, 106078 (2020).
    DOI: https://doi.org/10.1016/j.asoc.2020.106078

    This is a six-objective, unconstrained, continuous problem with 3 decision variables and 7 constraints.
    """

    def __init__(self):
        super(RE61, self).__init__()

        self.obj_directions = [self.MINIMIZE] * 6
        self.obj_labels = ["f(w)", "f(x)", "f(y)", "f(z)", "f(v)", "f(u)"]
        self.number_of_original_constraints = 7

        self.lower_bound = [0.01, 0.01, 0.01]
        self.upper_bound = [0.45, 0.10, 0.10]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = solution.variables[0]
        x2 = solution.variables[1]
        x3 = solution.variables[2]

        # Calculate the first five objectives
        solution.objectives[0] = 106780.37 * (x2 + x3) + 61704.67
        solution.objectives[1] = 3000 * x1
        solution.objectives[2] = 305700 * 2289 * x2 / (0.06 * 2289) ** 0.65
        solution.objectives[3] = 250 * 2289 * math.exp(-39.75 * x2 + 9.9 * x3 + 2.74)
        solution.objectives[4] = 25 * (1.39 / (x1 * x2) + 4940 * x3 - 80)

        # Constraint handling for the sixth objective
        g = [0.0] * self.number_of_original_constraints
        
        g[0] = 1 - (0.00139 / (x1 * x2) + 4.94 * x3 - 0.08)
        g[1] = 1 - (0.000306 / (x1 * x2) + 1.082 * x3 - 0.0986)
        g[2] = 50000 - (12.307 / (x1 * x2) + 49408.24 * x3 + 4051.02)
        g[3] = 16000 - (2.098 / (x1 * x2) + 8046.33 * x3 - 696.71)
        g[4] = 10000 - (2.138 / (x1 * x2) + 7883.39 * x3 - 705.04)
        g[5] = 2000 - (0.417 * x1 * x2 + 1721.26 * x3 - 136.54)
        g[6] = 550 - (0.164 / (x1 * x2) + 631.13 * x3 - 54.48)

        # Convert constraints to violations (only count when g[i] > 0)
        for i in range(self.number_of_original_constraints):
            g[i] = max(0.0, g[i])

        # Sixth objective (sum of constraint violations)
        solution.objectives[5] = sum(g)

        return solution

    def name(self) -> str:
        return "RE61"


class RE91(FloatProblem):
    """Problem RE91 from:
    Ryoji Tanabe and Hisao Ishibuchi, "An easy-to-use real-world multi-objective optimization
    problem suite", Applied Soft Computing, Vol. 89, 106078 (2020).
    DOI: https://doi.org/10.1016/j.asoc.2020.106078

    This is a nine-objective, unconstrained, continuous problem with 7 decision variables
    plus 4 random variables for a total of 11 variables.
    """

    def __init__(self):
        super(RE91, self).__init__()

        self.obj_directions = [self.MINIMIZE] * 9
        self.obj_labels = [f"f({i+1})" for i in range(9)]
        
        # Bounds for the first 7 variables
        self.lower_bound = [0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4]
        self.upper_bound = [1.5, 1.35, 1.5, 1.5, 2.265, 1.2, 1.2]
        
        # Add bounds for the random variables (will be set during evaluation)
        self.lower_bound.extend([-float('inf')] * 4)
        self.upper_bound.extend([float('inf')] * 4)
        
        # Initialize random number generator
        import random
        self.random = random.Random()

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0
        
    def create_solution(self) -> FloatSolution:
        solution = super().create_solution()
        # Initialize the random variables (indices 7-10) with 0.0, they'll be set in evaluate
        for i in range(7, 11):
            solution.variables[i] = 0.0
        return solution

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables.copy()
        
        # Set random variables (indices 7-10)
        x[7] = 0.006 * self.random.gauss(0, 1) + 0.345  # x7
        x[8] = 0.006 * self.random.gauss(0, 1) + 0.192  # x8
        x[9] = 10 * self.random.gauss(0, 1)             # x9
        x[10] = 10 * self.random.gauss(0, 1)            # x10

        # Calculate the nine objectives
        solution.objectives[0] = (
            1.98 + 4.9 * x[0] + 6.67 * x[1] + 6.98 * x[2] + 4.01 * x[3] + 
            1.75 * x[4] + 0.00001 * x[5] + 2.73 * x[6]
        )
        
        solution.objectives[1] = max(0.0, (
            1.16 - 0.3717 * x[1] * x[3] - 0.00931 * x[1] * x[9] - 
            0.484 * x[2] * x[8] + 0.01343 * x[5] * x[9]
        ) / 1.0)
        
        solution.objectives[2] = max(0.0, (
            0.261 - 0.0159 * x[0] * x[1] - 0.188 * x[0] * x[7] - 0.019 * x[1] * x[6] + 
            0.0144 * x[2] * x[4] + 0.87570001 * x[4] * x[9] + 0.08045 * x[5] * x[8] + 
            0.00139 * x[7] * x[10] + 0.00001575 * x[9] * x[10]
        ) / 0.32)
        
        solution.objectives[3] = max(0.0, (
            0.214 + 0.00817 * x[4] - 0.131 * x[0] * x[7] - 0.0704 * x[0] * x[8] + 
            0.03099 * x[1] * x[5] - 0.018 * x[1] * x[6] + 0.0208 * x[2] * x[7] + 
            0.121 * x[2] * x[8] - 0.00364 * x[4] * x[5] + 0.0007715 * x[4] * x[9] - 
            0.0005354 * x[5] * x[9] + 0.00121 * x[7] * x[10] + 0.00184 * x[8] * x[9] - 
            0.018 * x[1] * x[1]
        ) / 0.32)
        
        solution.objectives[4] = max(0.0, (
            0.74 - 0.61 * x[1] - 0.163 * x[2] * x[7] + 0.001232 * x[2] * x[9] - 
            0.166 * x[6] * x[8] + 0.227 * x[1] * x[1]
        ) / 0.32)
        
        # Objective 5 is more complex with multiple terms
        temp = (
            (28.98 + 3.818 * x[2] - 4.2 * x[0] * x[1] + 0.0207 * x[4] * x[9] + 
             6.63 * x[5] * x[8] - 7.77 * x[6] * x[7] + 0.32 * x[8] * x[9]) + 
            (33.86 + 2.95 * x[2] + 0.1792 * x[9] - 5.057 * x[0] * x[1] - 
             11 * x[1] * x[7] - 0.0215 * x[4] * x[9] - 9.98 * x[6] * x[7] + 
             22 * x[7] * x[8]) + 
            (46.36 - 9.9 * x[1] - 12.9 * x[0] * x[7] + 0.1107 * x[2] * x[9])
        ) / 3
        solution.objectives[5] = max(0.0, temp / 32)
        
        solution.objectives[6] = max(0.0, (
            4.72 - 0.5 * x[3] - 0.19 * x[1] * x[2] - 0.0122 * x[3] * x[9] + 
            0.009325 * x[5] * x[9] + 0.000191 * x[10] * x[10]
        ) / 4.0)
        
        solution.objectives[7] = max(0.0, (
            10.58 - 0.674 * x[0] * x[1] - 1.95 * x[1] * x[7] + 0.02054 * x[2] * x[9] - 
            0.0198 * x[3] * x[9] + 0.028 * x[5] * x[9]
        ) / 9.9)
        
        solution.objectives[8] = max(0.0, (
            16.45 - 0.489 * x[2] * x[6] - 0.843 * x[4] * x[5] + 0.0432 * x[8] * x[9] - 
            0.0556 * x[8] * x[10] - 0.000786 * x[10] * x[10]
        ) / 15.7)

        return solution

    def name(self) -> str:
        return "RE91"
