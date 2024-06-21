from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class Ahmad2017(FloatProblem):
    """
    Problem Ahmad2017 (RWA10) described in the paper "Engineering applications of multi-objective evolutionary algorithms:
    A test suite of box-constrained real-world problems". DOI: https://doi.org/10.1016/j.engappai.2023.106192
    """

    def __init__(self):
        super(Ahmad2017, self).__init__()

        self.lower_bound = [10.0, 10.0, 150.0]
        self.upper_bound = [50.0, 50.0, 170.0]

        self.obj_directions = [self.MAXIMIZE, self.MAXIMIZE, self.MAXIMIZE, self.MAXIMIZE, self.MINIMIZE, self.MAXIMIZE,
                               self.MAXIMIZE]
        self.obj_labels = ["WCA", "OCA", "AP", "CRA", "Stiffness", "Tear", "Tensile"]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = solution.variables[0]
        x2 = solution.variables[1]
        x3 = solution.variables[2]

        wca = -1331.04 + 1.99 * x1 + 0.33 * x2 + 17.12 * x3 - 0.02 * x1 * x1 - 0.05 * x3 * x3 - 15.33
        oca = -4231.14 + 4.27 * x1 + 1.50 * x2 + 52.30 * x3 - 0.04 * x1 * x2 - 0.04 * x1 * x1 - 0.16 * x3 * x3 - 29.33
        ap = 1766.80 - 32.32 * x1 - 24.56 * x2 - 10.48 * x3 + 0.24 * x1 * x3 + 0.19 * x2 * x3 - 0.06 * x1 * x1 - 0.10 * x2 * x2 - 413.33
        cra = -2342.13 - 1.556 * x1 + 0.77 * x2 + 31.14 * x3 + 0.03 * x1 * x1 - 0.10 * x3 * x3 - 73.33
        stiffness = 9.34 + 0.02 * x1 - 0.03 * x2 - 0.03 * x3 - 0.001 * x1 * x2 + 0.0009 * x2 * x2 + 0.22
        tear = 1954.71 + 14.246 * x1 + 5.00 * x2 - 4.30 * x3 - 0.22 * x1 * x1 - 0.33 * x2 * x2 - 8413.33
        tensile = 828.16 + 3.55 * x1 + 73.65 * x2 + 10.80 * x3 - 0.56 * x2 * x3 + 0.20 * x2 * x2 - 2814.83

        solution.objectives[0] = -wca  # maximization
        solution.objectives[1] = -oca  # maximization
        solution.objectives[2] = -ap  # maximization
        solution.objectives[3] = -cra  # maximization
        solution.objectives[4] = stiffness  # minimization
        solution.objectives[5] = -tear  # maximization
        solution.objectives[6] = -tensile  # maximization

        return solution

    def name(self):
        return "Ahmad2017"


class Chen2015(FloatProblem):
    """ Problem Chen2015 (RWA9) described in the paper "Engineering applications of multi-objective evolutionary
    algorithms: A test suite of box-constrained real-world problems". DOI: https://doi.org/10.1016/j.engappai.2023.106192
    """

    def __init__(self):
        super(Chen2015, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MAXIMIZE, self.MAXIMIZE, self.MAXIMIZE, self.MINIMIZE]
        self.obj_labels = ['F1', 'F2', 'F3', 'F4', 'F5']

        self.lower_bound = [17.5, 17.5, 2.0, 2.0, 5.0, 5.0]
        self.upper_bound = [22.5, 22.5, 3.0, 3.0, 7.0, 6.0]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        l1 = solution.variables[0]
        w1 = solution.variables[1]
        l2 = solution.variables[2]
        w2 = solution.variables[3]
        a1 = solution.variables[4]
        b1 = solution.variables[5]
        a2 = l1 * w1 * l2 * w2
        b2 = l1 * w1 * l2 * a1
        d2 = w1 * w2 * a1 * b1

        f1 = 502.94 - 27.18 * ((w1 - 20.0) / 0.5) + 43.08 * ((l1 - 20.0) / 2.5) + 47.75 * (a1 - 6.0) + 32.25 * (
                (b1 - 5.5) / 0.5) + 31.67 * (a2 - 11.0) - 36.19 * ((w1 - 20.0) / 0.5) * (
                     (w2 - 2.5) / 0.5) - 39.44 * ((w1 - 20.0) / 0.5) * (a1 - 6.0) + 57.45 * (a1 - 6.0) * (
                     (b1 - 5.5) / 0.5)
        f2 = 130.53 + 45.97 * ((l1 - 20.0) / 2.5) - 52.93 * ((w1 - 20.0) / 0.5) - 78.93 * (a1 - 6.0) + 79.22 * (
                a2 - 11.0) + 47.23 * ((w1 - 20.0) / 0.5) * (a1 - 6.0) - 40.61 * ((w1 - 20.0) / 0.5) * (
                     a2 - 11.0) - 50.62 * (a1 - 6.0) * (a2 - 11.0)
        f3 = 203.16 - 42.75 * ((w1 - 20.0) / 0.5) + 56.67 * (a1 - 6.0) + 19.88 * ((b1 - 5.5) / 0.5) - 12.89 * (
                a2 - 11.0) - 35.09 * (a1 - 6.0) * ((b1 - 5.5) / 0.5) - 22.91 * ((b1 - 5.5) / 0.5) * (a2 - 11.0)
        f4 = 0.76 - 0.06 * ((l1 - 20.0) / 2.5) + 0.03 * ((l2 - 2.5) / 0.5) + 0.02 * (a2 - 11.0) - 0.02 * (
                (b2 - 6.5) / 0.5) - 0.03 * ((d2 - 12.0) / 0.5) + 0.03 * ((l1 - 20.0) / 2.5) * (
                     (w1 - 20.0) / 0.5) - 0.02 * ((l1 - 20.0) / 2.5) * ((l2 - 2.5) / 0.5) + 0.02 * (
                     (l1 - 20.0) / 2.5) * ((b2 - 6.5) / 0.5)
        f5 = 1.08 - 0.12 * ((l1 - 20.0) / 2.5) - 0.26 * ((w1 - 20.0) / 0.5) - 0.05 * (a2 - 11.0) - 0.12 * (
                (b2 - 6.5) / 0.5) + 0.08 * (a1 - 6.0) * ((b2 - 6.5) / 0.5) + 0.07 * (a2 - 6.0) * (
                     (b2 - 5.5) / 0.5)

        solution.objectives[0] = f1
        solution.objectives[1] = -f2
        solution.objectives[2] = -f3
        solution.objectives[3] = -f4
        solution.objectives[4] = f5

        return solution

    def name(self):
        return "Chen2015"


class Ganesan2013(FloatProblem):
    """ Problem Ganesan2013 (RWA3) described in the paper "Engineering applications of multi-objective evolutionary
    algorithms: A test suite of box-constrained real-world problems". DOI: https://doi.org/10.1016/j.engappai.2023.106192
    """

    def __init__(self):
        super(Ganesan2013, self).__init__()

        self.lower_bound = [0.25, 10000.0, 600.0]
        self.upper_bound = [0.55, 20000.0, 1100.0]

        self.obj_directions = [self.MAXIMIZE, self.MAXIMIZE, self.MINIMIZE]
        self.obj_labels = ['HC4_conversion', 'CO_selectivity', 'H2_CO_ratio']

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        O2CH4 = solution.variables[0]
        GV = solution.variables[1]
        T = solution.variables[2]

        HC4_conversion = (-8.87e-6) * (86.74 + 14.6 * O2CH4 - 3.06 * GV + 18.82 * T + 3.14 * O2CH4 * GV
                                       - 6.91 * O2CH4 * O2CH4 - 13.31 * T * T)

        CO_selectivity = (-2.152e-9) * (39.46 + 5.98 * O2CH4 - 2.4 * GV + 13.06 * T + 2.5 * O2CH4 * GV
                                        + 1.64 * GV * T - 3.9 * O2CH4 * O2CH4 - 10.15 * T * T
                                        - 3.69 * GV * GV * O2CH4) + 45.7

        H2_CO_ratio = (4.425e-10) * (1.29 - 0.45 * T - 0.112 * O2CH4 * GV - 0.142 * T * GV
                                     + 0.109 * O2CH4 * O2CH4 + 0.405 * T * T
                                     + 0.167 * T * T * GV) + 0.18

        solution.objectives[0] = -HC4_conversion  # maximization
        solution.objectives[1] = -CO_selectivity  # maximization
        solution.objectives[2] = H2_CO_ratio  # minimization

        return solution

    def name(self):
        return "Ganesan2013"


class Gao2020(FloatProblem):
    """ Problem Gao2020 (RWA5) described in the paper "Engineering applications of multi-objective evolutionary
    algorithms: A test suite of box-constrained real-world problems". DOI: https://doi.org/10.1016/j.engappai.2023.106192
    """

    def __init__(self):
        super(Gao2020, self).__init__()

        self.lower_bound = [40.0, 0.35, 333.0, 20.0, 3000.0, 0.1, 308.0, 150.0, 0.1]
        self.upper_bound = [100.0, 0.5, 363.0, 40.0, 4000.0, 3.0, 328.0, 200.0, 2.0]

        self.obj_directions = [self.MINIMIZE, self.MAXIMIZE, self.MAXIMIZE]
        self.obj_labels = ['t_eff', 'Q_eff', 'Phi_ex']

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        A = solution.variables[0]
        B = solution.variables[1]
        C = solution.variables[2]
        D = solution.variables[3]
        E = solution.variables[4]
        F = solution.variables[5]
        G = solution.variables[6]
        H = solution.variables[7]
        J = solution.variables[8]

        t_eff = 171.33 + 23.25 * A - 8.61 * B - 59.85 * C - 66.12 * D - 15.29 * E \
                - 83.32 * F + 37.72 * G + 12.67 * H + 0.46 * J - 0.47 * A * B \
                - 0.30 * A * C - 6.22 * A * D - 0.62 * A * E - 42.48 * A * F \
                + 3.11 * A * G + 4.45 * A * H - 0.22 * A * J + 7.46 * B * C \
                + 3.28 * B * D + 1.28 * B * E + 1.02 * B * F - 4.02 * B * G \
                - 2.29 * B * H - 0.16 * B * J + 19.25 * C * D - 14.83 * C * E \
                + 5.07 * C * F - 37.61 * C * G - 9.11 * C * H - 0.32 * C * J \
                + 8.53 * D * E + 18.46 * D * F - 14.28 * D * G - 7.05 * D * H \
                - 0.24 * D * J + 2.05 * E * F + 15.73 * E * G - 0.77 * E * H \
                - 0.29 * E * J - 4.77 * F * G + 2.07 * F * H + 0.64 * F * J \
                + 3.41 * G * H + 1.76 * G * J + 0.48 * H * J + 3.64 * A * A \
                - 0.99 * B * B + 30.5 * C * C + 21.63 * D * D + 1.72 * E * E \
                + 72.42 * F * F + 11.2 * G * G + 1.86 * H * H - 0.79 * J * J

        Q_eff = 577.73 - 1.22 * A - 19.56 * B + 102.05 * C - 1.83 * D + 27.28 * E \
                + 2.52 * F + 5.43 * G + 37.48 * H + 0.45 * J + 2.94 * A * B \
                - 2.96 * A * C + 0.66 * A * D + 0.09 * A * E - 0.43 * A * F \
                + 0.12 * A * G - 0.43 * A * H - 0.7 * A * J + 8.05 * B * C \
                + 0.53 * B * D + 4.43 * B * E - 0.6 * B * F - 0.46 * B * G \
                - 4.97 * B * H + 0.046 * B * J + 0.42 * C * D + 6.03 * C * E \
                + 0.21 * C * F + 2.63 * C * G + 0.17 * C * H - 0.43 * C * J \
                + 6.34 * D * E + 6.36 * D * F + 0.19 * D * G - 0.22 * D * H \
                + 0.39 * D * J - 7.09 * E * F + 3.06 * E * G - 0.15 * E * H \
                + 0.68 * E * J - 0.2 * F * G + 0.14 * F * H + 0.88 * F * J \
                + 0.45 * G * H - 0.014 * G * J + 0.99 * H * J + 0.55 * A * A \
                - 4.97 * B * B - 0.47 * C * C - 0.91 * D * D - 2.08 * E * E \
                - 1.43 * F * F + 0.43 * G * G + 1.06 * H * H + 0.98 * J * J

        Phi_ex = 0.81 - 9.26e-3 * A + 0.014 * B - 0.029 * C - 7.69e-4 * D \
                 + 4.05e-3 * E + 0.029 * F + 0.075 * G - 0.012 * H \
                 - 1.04e-3 * J - 2.63e-3 * A * B + 1.34e-4 * A * C \
                 - 1.48e-3 * A * D - 7.04e-4 * A * E + 0.013 * A * F \
                 + 6.55e-4 * A * G - 9.71e-3 * A * H + 1.08e-3 * A * J \
                 + 2.54e-3 * B * C - 4.83e-4 * B * D + 9.63e-4 * B * E \
                 + 1.21e-3 * B * F - 7.02e-3 * B * G - 1.21e-3 * B * H \
                 + 1.94e-5 * B * J - 1.15e-3 * C * D + 3.60e-3 * C * E \
                 + 5.60e-3 * C * F - 0.026 * C * G - 4.01e-3 * C * H \
                 + 1.35e-3 * C * J - 6.93e-3 * D * E - 3.16e-3 * D * F \
                 - 2.38e-4 * D * G + 7.32e-4 * D * H + 4.69e-4 * D * J \
                 + 8.18e-3 * E * F - 5.74e-3 * E * G + 1.44e-4 * E * H \
                 - 9.95e-5 * E * J - 2.09e-3 * F * G - 65e-4 * F * H \
                 - 1.99e-3 * F * J + 4.95e-3 * G * H + 8.70e-4 * G * J \
                 + 4.55e-4 * H * J - 9.32e-4 * A * A - 7.61e-4 * B * B \
                 + 0.016 * C * C + 1.24e-3 * D * D + 9.61e-4 * E * E \
                 - 0.024 * F * F - 8.63e-3 * G * G - 1.90e-4 * H * H \
                 - 7.56e-4 * J * J

        solution.objectives[0] = t_eff  # Minimize
        solution.objectives[1] = -Q_eff  # Maximize
        solution.objectives[2] = -Phi_ex  # Maximize

        return solution

    def name(self):
        return "Gao2020"


class Goel2007(FloatProblem):
    """ Problem Gao2020 (RWA7) described in the paper "Engineering applications of multi-objective evolutionary
    algorithms: A test suite of box-constrained real-world problems". DOI: https://doi.org/10.1016/j.engappai.2023.106192
    """

    def __init__(self):
        super(Goel2007, self).__init__()

        number_of_variables = 4
        self.lower_bound = [0.0] * number_of_variables
        self.upper_bound = [1.0] * number_of_variables

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['Xcc', 'TFmax', 'TTmax']

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        a = solution.variables[0]
        DHA = solution.variables[1]
        DOA = solution.variables[2]
        OPTT = solution.variables[3]
        Xcc = 0.153 - 0.322 * a + 0.396 * DHA + 0.424 * DOA + 0.0226 * OPTT \
              + 0.175 * a * a + 0.0185 * DHA * a - 0.0701 * DHA * DHA \
              - 0.251 * DOA * a + 0.179 * DOA * DHA + 0.0150 * DOA * DOA \
              + 0.0134 * OPTT * a + 0.0296 * OPTT * DHA + 0.0752 * OPTT * DOA \
              + 0.0192 * OPTT * OPTT

        TFmax = 0.692 + 0.477 * a - 0.687 * DHA - 0.080 * DOA - 0.0650 * OPTT \
                - 0.167 * a * a - 0.0129 * DHA * a + 0.0796 * DHA * DHA \
                - 0.0634 * DOA * a - 0.0257 * DOA * DHA + 0.0877 * DOA * DOA \
                - 0.0521 * OPTT * a + 0.00156 * OPTT * DHA + 0.00198 * OPTT * DOA \
                + 0.0184 * OPTT * OPTT

        TTmax = 0.370 - 0.205 * a + 0.0307 * DHA + 0.108 * DOA + 1.019 * OPTT \
                - 0.135 * a * a + 0.0141 * DHA * a + 0.0998 * DHA * DHA \
                + 0.208 * DOA * a - 0.0301 * DOA * DHA - 0.226 * DOA * DOA \
                + 0.353 * OPTT * a - 0.0497 * OPTT * DOA - 0.423 * OPTT * OPTT \
                + 0.202 * DHA * a * a - 0.281 * DOA * a * a - 0.342 * DHA * DHA * a \
                - 0.245 * DHA * DHA * DOA + 0.281 * DOA * DOA * DHA \
                - 0.184 * OPTT * OPTT * a - 0.281 * DHA * a * DOA

        solution.objectives[0] = Xcc
        solution.objectives[1] = TFmax
        solution.objectives[2] = TTmax

        return solution

    def name(self):
        return "Goel2007"


class Liao2008(FloatProblem):
    """ Problem Liao2008 (RWA2) described in the paper "Engineering applications of multi-objective evolutionary
    algorithms: A test suite of box-constrained real-world problems". DOI: https://doi.org/10.1016/j.engappai.2023.106192
    """

    def __init__(self):
        super(Liao2008, self).__init__()

        number_of_variables = 5
        self.lower_bound = [1.0] * number_of_variables
        self.upper_bound = [3.0] * number_of_variables

        self.obj_labels = ['Mass', 'Ain', 'Intrusion']
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        t1 = solution.variables[0]
        t2 = solution.variables[1]
        t3 = solution.variables[2]
        t4 = solution.variables[3]
        t5 = solution.variables[4]
        Mass = 1640.2823 + 2.3573285 * t1 + 2.3220035 * t2 + 4.5688768 * t3 \
               + 7.7213633 * t4 + 4.4559504 * t5
        Ain = 6.5856 + 1.15 * t1 - 1.0427 * t2 + 0.9738 * t3 + 0.8364 * t4 \
              - 0.3695 * t1 * t4 + 0.0861 * t1 * t5 + 0.3628 * t2 * t4 \
              - 0.1106 * t1 * t1 - 0.3437 * t3 * t3 + 0.1764 * t4 * t4
        Intrusion = -0.0551 + 0.0181 * t1 + 0.1024 * t2 + 0.0421 * t3 \
                    - 0.0073 * t1 * t2 + 0.024 * t2 * t3 - 0.0118 * t2 * t4 \
                    - 0.0204 * t3 * t4 - 0.008 * t3 * t5 - 0.0241 * t2 * t2 \
                    + 0.0109 * t4 * t4

        solution.objectives[0] = Mass  # Minimization
        solution.objectives[1] = Ain  # Minimization
        solution.objectives[2] = Intrusion  # Minimization

        return solution

    def name(self):
        return "Liao2008"


class Padhi2016(FloatProblem):
    """ Problem Padhi2016 (RWA4) described in the paper "Engineering applications of multi-objective evolutionary
    algorithms: A test suite of box-constrained real-world problems". DOI: https://doi.org/10.1016/j.engappai.2023.106192
    """

    def __init__(self):
        super(Padhi2016, self).__init__()

        self.lower_bound = [1.0, 10.0, 850.0, 20.0, 4.0]
        self.upper_bound = [1.4, 26.0, 1650.0, 40.0, 8.0]

        self.obj_labels = ['CR', 'Ra', 'DD']
        self.obj_directions = [self.MAXIMIZE, self.MINIMIZE, self.MINIMIZE]

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
        CR = 1.74 + 0.42 * x1 - 0.27 * x2 + 0.087 * x3 - 0.19 * x4 + 0.18 * x5 \
             + 0.11 * x1 * x1 + 0.036 * x4 * x4 - 0.025 * x5 * x5 \
             + 0.044 * x1 * x2 + 0.034 * x1 * x4 + 0.17 * x1 * x5 \
             - 0.028 * x2 * x4 + 0.093 * x3 * x4 - 0.033 * x4 * x5

        Ra = 2.19 + 0.26 * x1 - 0.088 * x2 + 0.037 * x3 - 0.16 * x4 + 0.069 * x5 \
             + 0.036 * x1 * x1 + 0.11 * x1 * x3 - 0.077 * x1 * x4 \
             - 0.075 * x2 * x3 + 0.054 * x2 * x4 + 0.090 * x3 * x5 \
             + 0.041 * x4 * x5

        DD = 0.095 + 0.013 * x1 - 8.625 * 1e-003 * x2 - 5.458 * 1e-003 * x3 \
             - 0.012 * x4 + 1.462 * 1e-003 * x1 * x1 - 6.635 * 1e-004 * x2 * x2 \
             - 1.788 * 1e-003 * x4 * x4 - 0.011 * x1 * x2 \
             - 6.188 * 1e-003 * x1 * x3 + 8.937 * 1e-003 * x1 * x4 \
             - 4.563 * 1e-003 * x1 * x5 - 0.012 * x2 * x3 \
             - 1.063 * 1e-003 * x2 * x4 + 2.438 * 1e-003 * x2 * x5 \
             - 1.937 * 1e-003 * x3 * x4 - 1.188 * 1e-003 * x3 * x5 \
             - 3.312 * 1e-003 * x4 * x5

        solution.objectives[0] = -CR  # Maximization
        solution.objectives[1] = Ra  # Minimization
        solution.objectives[2] = DD  # Minimization

        return solution

    def name(self):
        return "Padhi2016"


class Subasi2016(FloatProblem):
    """ Problem Subasi2016 (RWA1) described in the paper "Engineering applications of multi-objective evolutionary
    algorithms: A test suite of box-constrained real-world problems". DOI: https://doi.org/10.1016/j.engappai.2023.106192
    """

    def __init__(self):
        super(Subasi2016, self).__init__()

        self.lower_bound = [20.0, 6.0, 20.0, 0.0, 8000.0]
        self.upper_bound = [60.0, 15.0, 40.0, 30.0, 25000.0]

        self.obj_labels = ['Nu', 'f']
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        H = solution.variables[0]
        t = solution.variables[1]
        Sy = solution.variables[2]
        theta = solution.variables[3]
        Re = solution.variables[4]

        Nu = 89.027 + 0.300 * H - 0.096 * t - 1.124 * Sy - 0.968 * theta \
             + 4.148 * 10e-3 * Re + 0.0464 * H * t - 0.0244 * H * Sy \
             + 0.0159 * H * theta + 4.151 * 10e-5 * H * Re + 0.1111 * t * Sy \
             - 4.121 * 10e-5 * Sy * Re + 4.192 * 10e-5 * theta * Re

        f = 0.4753 - 0.0181 * H + 0.0420 * t + 5.481 * 10e-3 * Sy - 0.0191 * theta \
            - 3.416 * 10e-6 * Re - 8.851 * 10e-4 * H * Sy \
            + 8.702 * 10e-4 * H * theta + 1.536 * 10e-3 * t * theta \
            - 2.761 * 10e-6 * t * Re - 4.400 * 10e-4 * Sy * theta \
            + 9.714 * 10e-7 * Sy * Re + 6.777 * 10e-4 * H * H

        solution.objectives[0] = -Nu
        solution.objectives[1] = f

        return solution

    def name(self):
        return "Subasi2016"


class Vaidyanathan2004(FloatProblem):
    """ Problem Vaidyanathan2004 (RWA8) described in the paper "Engineering applications of multi-objective evolutionary
    algorithms: A test suite of box-constrained real-world problems". DOI: https://doi.org/10.1016/j.engappai.2023.106192
    """

    def __init__(self):
        super(Vaidyanathan2004, self).__init__()

        self.lower_bound = [0.0, 0.0, 0.0, 0.0]
        self.upper_bound = [1.0, 1.0, 1.0, 1.0]

        self.obj_labels = ['TFmax', 'TW4', 'TTmax', 'Xcc']
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        a = solution.variables[0]
        DHA = solution.variables[1]
        DOA = solution.variables[2]
        OPTT = solution.variables[3]

        TFmax = 0.692 + 0.477 * a - 0.687 * DHA - 0.080 * DOA - 0.0650 * OPTT \
                - 0.167 * a * a - 0.0129 * DHA * a + 0.0796 * DHA * DHA \
                - 0.0634 * DOA * a - 0.0257 * DOA * DHA + 0.0877 * DOA * DOA \
                - 0.0521 * OPTT * a + 0.00156 * OPTT * DHA + 0.00198 * OPTT * DOA \
                + 0.0184 * OPTT * OPTT

        Xcc = 0.153 - 0.322 * a + 0.396 * DHA + 0.424 * DOA + 0.0226 * OPTT \
              + 0.175 * a * a + 0.0185 * DHA * a - 0.0701 * DHA * DHA \
              - 0.251 * DOA * a + 0.179 * DOA * DHA + 0.0150 * DOA * DOA \
              + 0.0134 * OPTT * a + 0.0296 * OPTT * DHA + 0.0752 * OPTT * DOA \
              + 0.0192 * OPTT * OPTT

        TW4 = 0.758 + 0.358 * a - 0.807 * DHA + 0.0925 * DOA - 0.0468 * OPTT \
              - 0.172 * a * a + 0.0106 * DHA * a + 0.0697 * DHA * DHA \
              - 0.146 * DOA * a - 0.0416 * DOA * DHA + 0.102 * DOA * DOA \
              - 0.0694 * OPTT * a - 0.00503 * OPTT * DHA + 0.0151 * OPTT * DOA \
              + 0.0173 * OPTT * OPTT

        TTmax = 0.370 - 0.205 * a + 0.0307 * DHA + 0.108 * DOA + 1.019 * OPTT \
                - 0.135 * a * a + 0.0141 * DHA * a + 0.0998 * DHA * DHA \
                + 0.208 * DOA * a - 0.0301 * DOA * DHA - 0.226 * DOA * DOA \
                + 0.353 * OPTT * a - 0.0497 * OPTT * DOA - 0.423 * OPTT * OPTT \
                + 0.202 * DHA * a * a - 0.281 * DOA * a * a - 0.342 * DHA * DHA * a \
                - 0.245 * DHA * DHA * DOA + 0.281 * DOA * DOA * DHA \
                - 0.184 * OPTT * OPTT * a - 0.281 * DHA * a * DOA

        solution.objectives[0] = TFmax
        solution.objectives[1] = TW4
        solution.objectives[2] = TTmax
        solution.objectives[3] = Xcc

        return solution

    def name(self):
        return "Vaidyanathan2004"


class Xu2020(FloatProblem):
    """ Problem Xu2020 (RWA6) described in the paper "Engineering applications of multi-objective evolutionary
    algorithms: A test suite of box-constrained real-world problems". DOI: https://doi.org/10.1016/j.engappai.2023.106192
    """
    def __init__(self):
        super(Xu2020, self).__init__()

        self.lower_bound = [12.56, 0.02, 1.0, 0.5]
        self.upper_bound = [25.12, 0.06, 5.0, 2.0]

        self.obj_labels = ['Ft', 'Ra', 'MRR']
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MAXIMIZE]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        vc = solution.variables[0]
        fz = solution.variables[1]
        ap = solution.variables[2]
        ae = solution.variables[3]

        d = 2.5
        z = 1.0

        Ft = -54.3 - 1.18 * vc - 2429 * fz + 104.2 * ap + 129.0 * ae \
             - 18.9 * vc * fz - 0.209 * vc * ap - 0.673 * vc * ae + 265 * fz * ap \
             + 1209 * fz * ae + 22.76 * ap * ae + 0.066 * vc * vc \
             + 32117 * fz * fz - 16.98 * ap * ap - 47.6 * ae * ae

        Ra = 0.227 - 0.0072 * vc + 1.89 * fz - 0.0203 * ap + 0.3075 * ae \
             - 0.198 * vc * fz - 0.000955 * vc * ap - 0.00656 * vc * ae \
             + 0.209 * fz * ap + 0.783 * fz * ae + 0.02275 * ap * ae \
             + 0.000355 * vc * vc + 35 * fz * fz + 0.00037 * ap * ap \
             - 0.0791 * ae * ae

        MRR = (1000.0 * vc * fz * z * ap * ae) / (3.14159 * d)

        solution.objectives[0] = Ft
        solution.objectives[1] = Ra
        solution.objectives[2] = -MRR

        return solution

    def name(self):
        return "Xu2020"
