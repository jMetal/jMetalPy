from math import sin, pi, cos, sqrt

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class LIRCMOP1(FloatProblem):
    """ Class representing problem LIR-CMOP1, defined in:

    * An Improved epsilon-constrained Method in MOEA/D for CMOPs with Large Infeasible Regions.
      Fan, Z., Li, W., Cai, X. et al. Soft Comput (2019). https://doi.org/10.1007/s00500-019-03794-x
    """

    def __init__(self, number_of_variables: int = 30):
        super(LIRCMOP1, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 2

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = [0.0 for _ in range(self.number_of_variables)]
        self.upper_bound = [1.0 for _ in range(self.number_of_variables)]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables

        solution.objectives[0] = x[0] + self.g1(x)
        solution.objectives[1] = 1 - x[0] * x[0] + self.g2(x)

        self.evaluate_constraints(solution)

        return solution

    def evaluate_constraints(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        constraints = [0.0 for _ in range(self.number_of_constraints)]

        a = 0.51
        b = 0.5

        constraints[0] = (a - self.g1(x)) * (self.g1(x) - b)
        constraints[1] = (a - self.g2(x)) * (self.g2(x) - b)

        solution.constraints = constraints

        return solution

    def g1(self, x: [float]) -> float:
        result = 0
        for i in range(2, self.number_of_variables, 2):
            result += pow(x[i] - sin(0.5 * pi * x[0]), 2.0)

        return result

    def g2(self, x: [float]) -> float:
        result = 0
        for i in range(1, self.number_of_variables, 2):
            result += pow(x[i] - cos(0.5 * pi * x[0]), 2.0)

        return result

    def get_name(self):
        return 'LIR-CMOP1'


class LIRCMOP2(LIRCMOP1):
    """ Class representing problem LIR-CMOP1, defined in:

    * An Improved epsilon-constrained Method in MOEA/D for CMOPs with Large Infeasible Regions.
      Fan, Z., Li, W., Cai, X. et al. Soft Comput (2019). https://doi.org/10.1007/s00500-019-03794-x
    """

    def __init__(self, number_of_variables: int = 30):
        super(LIRCMOP2, self).__init__(number_of_variables)

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables

        solution.objectives[0] = x[0] + self.g1(x)
        solution.objectives[1] = 1 - sqrt(x[0]) + self.g2(x)

        self.evaluate_constraints(solution)

        return solution

    def get_name(self):
        return 'LIR-CMOP2'


class LIRCMOP3(LIRCMOP1):
    """ Class representing problem LIR-CMOP3, defined in:

    * An Improved epsilon-constrained Method in MOEA/D for CMOPs with Large Infeasible Regions.
      Fan, Z., Li, W., Cai, X. et al. Soft Comput (2019). https://doi.org/10.1007/s00500-019-03794-x
    """

    def __init__(self, number_of_variables: int = 30):
        super(LIRCMOP3, self).__init__(number_of_variables)

    def evaluate_constraints(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        constraints = [0.0 for _ in range(self.number_of_constraints)]

        a = 0.51
        b = 0.5
        c = 20.0

        constraints[0] = (a - self.g1(x)) * (self.g1(x) - b)
        constraints[1] = (a - self.g2(x)) * (self.g2(x) - b)
        constraints[2] = sin(c * pi * x[0]) - 0.5

        solution.constraints = constraints

        return solution

    def get_name(self):
        return 'LIR-CMOP3'


class LIRCMOP4(LIRCMOP2):
    """ Class representing problem LIR-CMOP4, defined in:

    * An Improved epsilon-constrained Method in MOEA/D for CMOPs with Large Infeasible Regions.
      Fan, Z., Li, W., Cai, X. et al. Soft Comput (2019). https://doi.org/10.1007/s00500-019-03794-x
    """

    def __init__(self, number_of_variables: int = 30):
        super(LIRCMOP4, self).__init__(number_of_variables)

    def evaluate_constraints(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        constraints = [0.0 for _ in range(self.number_of_constraints)]

        a = 0.51
        b = 0.5
        c = 20.0

        constraints[0] = (a - self.g1(x)) * (self.g1(x) - b)
        constraints[1] = (a - self.g2(x)) * (self.g2(x) - b)
        constraints[2] = sin(c * pi * x[0]) - 0.5

        solution.constraints = constraints

        return solution

    def get_name(self):
        return 'LIR-CMOP4'


class LIRCMOP5(FloatProblem):
    """ Class representing problem LIR-CMOP5, defined in:

    * An Improved epsilon-constrained Method in MOEA/D for CMOPs with Large Infeasible Regions.
      Fan, Z., Li, W., Cai, X. et al. Soft Comput (2019). https://doi.org/10.1007/s00500-019-03794-x
    """

    def __init__(self, number_of_variables: int = 30):
        super(LIRCMOP5, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 2

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = [0.0 for _ in range(self.number_of_variables)]
        self.upper_bound = [1.0 for _ in range(self.number_of_variables)]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables

        solution.objectives[0] = x[0] + 10 * self.g1(x) + 0.7057
        solution.objectives[1] = 1 - sqrt(x[0]) + 10 * self.g2(x) + 7057

        self.evaluate_constraints(solution)

        return solution

    def evaluate_constraints(self, solution: FloatSolution) -> FloatSolution:
        constraints = [0.0 for _ in range(self.number_of_constraints)]

        r = 0.1
        theta = -0.25 * pi
        a_array = [2.0, 2.0]
        b_array = [4.0, 8.0]
        x_offset = [1.6, 2.5]
        y_offset = [1.6, 2.5]
        f1 = solution.objectives[0]
        f2 = solution.objectives[1]

        for i in range(len(x_offset)):
            constraints[i] = pow(
                ((f1 - x_offset[i]) * cos(theta) - (f2 - y_offset[i]) * sin(theta)) / a_array[i], 2) + \
                             pow(
                                 ((f1 - x_offset[i]) * sin(theta) + (f2 - y_offset[i]) * cos(theta)) / b_array[i],
                                 2) - r

        solution.constraints = constraints

        return solution

    def g1(self, x: [float]) -> float:
        result = 0
        for i in range(2, self.number_of_variables, 2):
            result += pow(x[i] - sin(0.5 * i / len(x) * pi * x[0]), 2.0)

        return result

    def g2(self, x: [float]) -> float:
        result = 0
        for i in range(1, self.number_of_variables, 2):
            result += pow(x[i] - cos(0.5 * i / len(x) * pi * x[0]), 2.0)

        return result

    def get_name(self):
        return 'LIR-CMOP5'


class LIRCMOP6(LIRCMOP5):
    """ Class representing problem LIR-CMOP6, defined in:

    * An Improved epsilon-constrained Method in MOEA/D for CMOPs with Large Infeasible Regions.
      Fan, Z., Li, W., Cai, X. et al. Soft Comput (2019). https://doi.org/10.1007/s00500-019-03794-x
    """

    def __init__(self, number_of_variables: int = 30):
        super(LIRCMOP6, self).__init__(number_of_variables)

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables

        solution.objectives[0] = x[0] + 10 * self.g1(x) + 0.7057
        solution.objectives[1] = 1 - x[0] * x[0] + 10 * self.g2(x) + 7057

        self.evaluate_constraints(solution)

        return solution

    def evaluate_constraints(self, solution: FloatSolution) -> FloatSolution:
        constraints = [0.0 for _ in range(self.number_of_constraints)]

        r = 0.1
        theta = -0.25 * pi
        a_array = [2.0, 2.0]
        b_array = [8.0, 8.0]
        x_offset = [1.8, 2.8]
        y_offset = [1.8, 2.8]
        f1 = solution.objectives[0]
        f2 = solution.objectives[1]

        for i in range(len(x_offset)):
            constraints[i] = pow(
                ((f1 - x_offset[i]) * cos(theta) - (f2 - y_offset[i]) * sin(theta)) / a_array[i], 2) + \
                             pow(
                                 ((f1 - x_offset[i]) * sin(theta) + (f2 - y_offset[i]) * cos(theta)) / b_array[i],
                                 2) - r

        solution.constraints = constraints

        return solution

    def get_name(self):
        return 'LIR-CMOP6'


class LIRCMOP7(LIRCMOP5):
    """ Class representing problem LIR-CMOP7, defined in:

    * An Improved epsilon-constrained Method in MOEA/D for CMOPs with Large Infeasible Regions.
      Fan, Z., Li, W., Cai, X. et al. Soft Comput (2019). https://doi.org/10.1007/s00500-019-03794-x
    """

    def __init__(self, number_of_variables: int = 30):
        super(LIRCMOP7, self).__init__(number_of_variables)

    def evaluate_constraints(self, solution: FloatSolution) -> FloatSolution:
        constraints = [0.0 for _ in range(self.number_of_constraints)]

        r = 0.1
        theta = -0.25 * pi
        a_array = [2.0, 2.5, 2.5]
        b_array = [6.0, 12.0, 10.0]
        x_offset = [1.2, 2.25, 3.5]
        y_offset = [1.2, 2.25, 3.5]
        f1 = solution.objectives[0]
        f2 = solution.objectives[1]

        for i in range(len(x_offset)):
            constraints[i] = pow(
                ((f1 - x_offset[i]) * cos(theta) - (f2 - y_offset[i]) * sin(theta)) / a_array[i], 2) + \
                             pow(
                                 ((f1 - x_offset[i]) * sin(theta) + (f2 - y_offset[i]) * cos(theta)) / b_array[i],
                                 2) - r

        solution.constraints = constraints

        return solution

    def get_name(self):
        return 'LIR-CMOP7'


class LIRCMOP8(LIRCMOP6):
    """ Class representing problem LIR-CMOP8, defined in:

    * An Improved epsilon-constrained Method in MOEA/D for CMOPs with Large Infeasible Regions.
      Fan, Z., Li, W., Cai, X. et al. Soft Comput (2019). https://doi.org/10.1007/s00500-019-03794-x
    """

    def __init__(self, number_of_variables: int = 30):
        super(LIRCMOP8, self).__init__(number_of_variables)

    def evaluate_constraints(self, solution: FloatSolution) -> FloatSolution:
        constraints = [0.0 for _ in range(self.number_of_constraints)]

        r = 0.1
        theta = -0.25 * pi
        a_array = [2.0, 2.5, 2.5]
        b_array = [6.0, 12.0, 10.0]
        x_offset = [1.2, 2.25, 3.5]
        y_offset = [1.2, 2.25, 3.5]
        f1 = solution.objectives[0]
        f2 = solution.objectives[1]

        for i in range(len(x_offset)):
            constraints[i] = pow(
                ((f1 - x_offset[i]) * cos(theta) - (f2 - y_offset[i]) * sin(theta)) / a_array[i], 2) + \
                             pow(
                                 ((f1 - x_offset[i]) * sin(theta) + (f2 - y_offset[i]) * cos(theta)) / b_array[i],
                                 2) - r

        solution.constraints = constraints

        return solution

    def get_name(self):
        return 'LIR-CMOP8'


class LIRCMOP9(LIRCMOP8):
    """ Class representing problem LIR-CMOP9, defined in:

    * An Improved epsilon-constrained Method in MOEA/D for CMOPs with Large Infeasible Regions.
      Fan, Z., Li, W., Cai, X. et al. Soft Comput (2019). https://doi.org/10.1007/s00500-019-03794-x
    """

    def __init__(self, number_of_variables: int = 30):
        super(LIRCMOP9, self).__init__(number_of_variables)

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables

        solution.objectives[0] = 1.7057 * x[0] * (10 * self.g1(x) + 1)
        solution.objectives[1] = 1.7957 * (1 - x[0] * x[0]) * (10 * self.g2(x) + 1)

        self.evaluate_constraints(solution)

        return solution

    def evaluate_constraints(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        constraints = [0.0 for _ in range(self.number_of_constraints)]

        theta = -0.25 * pi
        n = 4.0
        f0 = solution.objectives[0]
        f1 = solution.objectives[1]

        constraints[0] = f0 * sin(theta) + f1 * cos(theta) - sin(n * pi * (f0 * cos(theta) - f1 * sin(theta))) - 2;

        x_offset = 1.40
        y_offset = 1.40
        a = 1.5
        b = 6.0
        r = 0.1;

        constraints[1] = pow(((f0 - x_offset) * cos(theta) - (f1 - y_offset) * sin(theta)) / a, 2) + pow(
            ((f0 - x_offset) * sin(theta) + (f1 - y_offset) * cos(theta)) / b, 2) - r

        solution.constraints = constraints

        return solution

    def get_name(self):
        return 'LIR-CMOP9'


class LIRCMOP10(LIRCMOP8):
    """ Class representing problem LIR-CMOP10, defined in:

    * An Improved epsilon-constrained Method in MOEA/D for CMOPs with Large Infeasible Regions.
      Fan, Z., Li, W., Cai, X. et al. Soft Comput (2019). https://doi.org/10.1007/s00500-019-03794-x
    """

    def __init__(self, number_of_variables: int = 30):
        super(LIRCMOP10, self).__init__(number_of_variables)

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables

        solution.objectives[0] = 1.7057 * x[0] * (10 * self.g1(x) + 1)
        solution.objectives[1] = 1.7957 * (1 - sqrt(x[0])) * (10 * self.g2(x) + 1)

        self.evaluate_constraints(solution)

        return solution

    def evaluate_constraints(self, solution: FloatSolution) -> FloatSolution:
        constraints = [0.0 for _ in range(self.number_of_constraints)]

        theta = -0.25 * pi
        n = 4.0
        f0 = solution.objectives[0]
        f1 = solution.objectives[1]

        constraints[0] = f0 * sin(theta) + f1 * cos(theta) - sin(n * pi * (f0 * cos(theta) - f1 * sin(theta))) - 1;

        x_offset = 1.1
        y_offset = 1.2
        a = 2.0
        b = 4.0
        r = 0.1;

        constraints[1] = pow(((f0 - x_offset) * cos(theta) - (f1 - y_offset) * sin(theta)) / a, 2) + pow(
            ((f0 - x_offset) * sin(theta) + (f1 - y_offset) * cos(theta)) / b, 2) - r

        solution.constraints = constraints

        return solution

    def get_name(self):
        return 'LIR-CMOP10'


class LIRCMOP11(LIRCMOP10):
    """ Class representing problem LIR-CMOP11, defined in:

    * An Improved epsilon-constrained Method in MOEA/D for CMOPs with Large Infeasible Regions.
      Fan, Z., Li, W., Cai, X. et al. Soft Comput (2019). https://doi.org/10.1007/s00500-019-03794-x
    """

    def __init__(self, number_of_variables: int = 30):
        super(LIRCMOP11, self).__init__(number_of_variables)

    def evaluate_constraints(self, solution: FloatSolution) -> FloatSolution:
        constraints = [0.0 for _ in range(self.number_of_constraints)]

        theta = -0.25 * pi
        n = 4.0
        f0 = solution.objectives[0]
        f1 = solution.objectives[1]

        constraints[0] = f0 * sin(theta) + f1 * cos(theta) - sin(n * pi * (f0 * cos(theta) - f1 * sin(theta))) - 2.1;

        x_offset = 1.2
        y_offset = 1.2
        a = 1.5
        b = 5.0
        r = 0.1;

        constraints[1] = pow(((f0 - x_offset) * cos(theta) - (f1 - y_offset) * sin(theta)) / a, 2) + pow(
            ((f0 - x_offset) * sin(theta) + (f1 - y_offset) * cos(theta)) / b, 2) - r

        solution.constraints = constraints

        return solution

    def get_name(self):
        return 'LIR-CMOP11'


class LIRCMOP12(LIRCMOP9):
    """ Class representing problem LIR-CMOP9, defined in:

    * An Improved epsilon-constrained Method in MOEA/D for CMOPs with Large Infeasible Regions.
      Fan, Z., Li, W., Cai, X. et al. Soft Comput (2019). https://doi.org/10.1007/s00500-019-03794-x
    """

    def __init__(self, number_of_variables: int = 30):
        super(LIRCMOP12, self).__init__(number_of_variables)

    def evaluate_constraints(self, solution: FloatSolution) -> FloatSolution:
        constraints = [0.0 for _ in range(self.number_of_constraints)]

        theta = -0.25 * pi
        n = 4.0
        f0 = solution.objectives[0]
        f1 = solution.objectives[1]

        constraints[0] = f0 * sin(theta) + f1 * cos(theta) - sin(n * pi * (f0 * cos(theta) - f1 * sin(theta))) - 2.5;

        x_offset = 1.6
        y_offset = 1.6
        a = 1.5
        b = 6.0
        r = 0.1;

        constraints[1] = pow(((f0 - x_offset) * cos(theta) - (f1 - y_offset) * sin(theta)) / a, 2) + pow(
            ((f0 - x_offset) * sin(theta) + (f1 - y_offset) * cos(theta)) / b, 2) - r

        solution.constraints = constraints

        return solution

    def get_name(self):
        return 'LIR-CMOP12'


class LIRCMOP13(FloatProblem):
    """ Class representing problem LIR-CMOP13, defined in:

    * An Improved epsilon-constrained Method in MOEA/D for CMOPs with Large Infeasible Regions.
      Fan, Z., Li, W., Cai, X. et al. Soft Comput (2019). https://doi.org/10.1007/s00500-019-03794-x
    """

    def __init__(self, number_of_variables: int = 30):
        super(LIRCMOP13, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 3
        self.number_of_constraints = 2

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = [0.0 for _ in range(self.number_of_variables)]
        self.upper_bound = [1.0 for _ in range(self.number_of_variables)]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables

        solution.objectives[0] = (1.7057 + self.g1(x)) * cos(0.5 * pi * x[0]) * cos(0.5 * pi + x[1])
        solution.objectives[1] = (1.7057 + self.g1(x)) * cos(0.5 * pi * x[0]) * sin(0.5 * pi + x[1])
        solution.objectives[2] = (1.7057 + self.g1(x)) * sin(0.5 * pi + x[0])

        self.evaluate_constraints(solution)

        return solution

    def evaluate_constraints(self, solution: FloatSolution) -> FloatSolution:
        constraints = [0.0 for _ in range(self.number_of_constraints)]

        f = sum([pow(solution.objectives[i], 2) for i in range(solution.number_of_objectives)])

        constraints[0] = (f - 9) * (f - 4)
        constraints[1] = (f - 1.9 * 1.9) * (f - 1.8 * 1.8)

        solution.constraints = constraints

        return solution

    def g1(self, x: [float]) -> float:
        result = 0
        for i in range(2, self.number_of_variables, 2):
            result += 10 * pow(x[i] - 0.5, 2.0)

        return result

    def get_name(self):
        return 'LIR-CMOP13'


class LIRCMOP14(LIRCMOP13):
    """ Class representing problem LIR-CMOP14, defined in:

    * An Improved epsilon-constrained Method in MOEA/D for CMOPs with Large Infeasible Regions.
      Fan, Z., Li, W., Cai, X. et al. Soft Comput (2019). https://doi.org/10.1007/s00500-019-03794-x
    """

    def __init__(self, number_of_variables: int = 30):
        super(LIRCMOP14, self).__init__(number_of_variables)
        self.number_of_constraints = 3

    def evaluate_constraints(self, solution: FloatSolution) -> FloatSolution:
        constraints = [0.0 for _ in range(self.number_of_constraints)]

        f = sum([pow(solution.objectives[i], 2) for i in range(solution.number_of_objectives)])

        constraints[0] = (f - 9) * (f - 4)
        constraints[1] = (f - 1.9 * 1.9) * (f - 1.8 * 1.8)
        constraints[2] = (f - 1.75 * 1.75) * (f - 1.6 * 1.6)

        solution.constraints = constraints

        return solution

    def get_name(self):
        return 'LIR-CMOP14'
