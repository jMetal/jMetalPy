import math
from abc import ABCMeta

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

"""
.. module:: LZ09
   :platform: Unix, Windows
   :synopsis: LZ09 problem family of multi-objective problems.

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class LZ09(FloatProblem):

    __metaclass__ = ABCMeta

    def __init__(self,
                 number_of_variables: int,
                 number_of_objectives: int,
                 number_of_constraints: int,
                 ptype: int,
                 dtype: int,
                 ltype: int):
        """ LZ09 benchmark family as defined in:

        * H. Li and Q. Zhang. Multiobjective optimization problems with complicated pareto sets, MOEA/D and NSGA-II.
        IEEE Transactions on Evolutionary Computation, 12(2):284-302, April 2009.
        """
        super(LZ09, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = number_of_constraints

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)', 'f(z)']

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

        self.ptype = ptype
        self.dtype = dtype
        self.ltype = ltype

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        y = self.objective(x)

        for i in range(self.number_of_objectives):
            solution.objectives[i] = y[i]

        return solution

    def __ps_func2(self, x: float, t1: float, dim: int, type: int, css: int) -> float:
        """ Control the PS shapes of 2-D instances.

        :param type: The type of the curve.
        :param css: The class of the index.
        """
        beta = 0.0
        dim += 1

        if type == 21:
            xy = 2 * (x - 0.5)
            beta = xy - math.pow(t1, 0.5 * (self.number_of_variables + 3 * dim - 8) / (self.number_of_variables - 2))
        if type == 22:
            theta = 6 * math.pi * t1 + dim * math.pi / self.number_of_variables
            xy = 2 * (x - 0.5)
            beta = xy - math.sin(theta)
        if type == 23:
            theta = 6 * math.pi * t1 + dim * math.pi / self.number_of_variables
            ra = 0.8 * t1
            xy = 2 * (x - 0.5)
            if css == 1:
                beta = xy - ra * math.cos(theta)
            else:
                beta = xy - ra * math.sin(theta)
        if type == 24:
            theta = 6 * math.pi * t1 + dim * math.pi / self.number_of_variables
            xy = 2 * (x - 0.5)
            ra = 0.8 * t1
            if css == 1:
                beta = xy - ra * math.cos(theta / 3)
            else:
                beta = xy - ra * math.sin(theta)
        if type == 25:
            rho = 0.8
            phi = math.pi * t1
            theta = 6 * math.pi * t1 + dim * math.pi / self.number_of_variables
            xy = 2 * (x - 0.5)
            if css == 1:
                beta = xy - rho * math.sin(phi) * math.sin(theta)
            elif css == 2:
                beta = xy - rho * math.sin(phi) * math.cos(theta)
            else:
                beta = xy - rho * math.cos(phi)
        if type == 26:
            theta = 6 * math.pi * t1 + dim * math.pi / self.number_of_variables
            ra = 0.3 * t1 * (t1 * math.cos(4 * theta) + 2)
            xy = 2 * (x - 0.5)
            if css == 1:
                beta = xy - ra * math.cos(theta)
            else:
                beta = xy - ra * math.sin(theta)

        return beta

    def __ps_func3(self, x: float, t1: float, t2: float, dim: int, type: int):
        """ Control the PS shapes of 3-D instances.
        :param type: The type of curve.
        """
        beta = 0.0
        dim += 1

        if type == 31:
            xy = 4 * (x - .5)
            rate = 1.0 * dim / self.number_of_variables
            beta = xy - 4 * (t1 * t1 * rate + t2 * (1.0 - rate)) + 2
        if type == 32:
            theta = 2 * math.pi * t1 + dim * math.pi / self.number_of_variables
            xy = 4 * (x - .5)
            beta = xy - 2 * t2 * math.sin(theta)

        return beta

    def __alpha_func(self, x: list, dim: int, type: int) -> list:
        """ Control the PF shape.
        """
        alpha = [0.0] * dim

        if dim == 2:
            if type == 21:
                alpha[0] = x[0]
                alpha[1] = 1 - math.sqrt(x[0])
            if type == 22:
                alpha[0] = x[0]
                alpha[1] = 1 - x[0] * x[0]
            if type == 23:
                alpha[0] = x[0]
                alpha[1] = 1 - math.sqrt(alpha[0]) - alpha[0] * math.sin(10 * alpha[0] * alpha[0] * math.pi)
            if type == 24:
                alpha[0] = x[0]
                alpha[1] = 1 - x[0] - 0.05 * math.sin(4 * math.pi * x[0])
        else:
            if type == 31:
                alpha[0] = math.cos(x[0] * math.pi / 2) * math.cos(x[1] * math.pi / 2)
                alpha[1] = math.cos(x[0] * math.pi / 2) * math.sin(x[1] * math.pi / 2)
                alpha[2] = math.sin(x[0] * math.pi / 2)
            if type == 32:
                alpha[0] = 1 - math.cos(x[0] * math.pi / 2) * math.cos(x[1] * math.pi / 2)
                alpha[1] = 1 - math.cos(x[0] * math.pi / 2) * math.sin(x[1] * math.pi / 2)
                alpha[2] = 1 - math.sin(x[0] * math.pi / 2)
            if type == 33:
                alpha[0] = x[0]
                alpha[1] = x[1]
                alpha[2] = 3 - (math.sin(3 * math.pi * x[0]) + math.sin(3 * math.pi * x[1]) - 2 * (x[0] + x[1]))
            if type == 34:
                alpha[0] = x[0] - x[1]
                alpha[1] = x[0] * (1 - x[1])
                alpha[2] = 1 - x[0]

        return alpha

    def __beta_func(self, x: list, type: int) -> float:
        """ Control the distance.
        """
        beta = 0.0
        dim = len(x)

        if dim == 0:
            beta = 0.0
        if type == 1:
            for i in range(dim):
                beta += x[i] * x[i]
            beta = 2.0 * beta / dim
        if type == 2:
            for i in range(dim):
                beta += math.sqrt(i + 1) * x[i] * x[i]
            beta = 2.0 * beta / dim
        if type == 3:
            sum, xx = 0, 0
            for i in range(dim):
                xx = 2 * x[i]
                sum += (xx * xx - math.cos(4 * math.pi * xx) + 1)
            beta = 2.0 * sum / dim
        if type == 4:
            sum, prod, xx = 0, 1, 0
            for i in range(dim):
                xx = 2 * x[i]
                sum += xx * xx
                prod *= math.cos(10 * math.pi * xx / math.sqrt(i + 1))
            beta = 2.0 * (sum - 2 * prod + 2) / dim

        return beta

    def objective(self, x_variables: list) -> list:
        aa = []
        bb = []
        cc = []

        y_objectives = [0.0] * self.number_of_objectives

        if self.number_of_objectives == 2:
            if self.ltype in [21, 22, 23, 24, 26]:
                for n in range(1, self.number_of_variables):
                    if n % 2 == 0:
                        a = self.__ps_func2(x_variables[n], x_variables[0], n, self.ltype, 1)
                        aa.append(a)
                    else:
                        b = self.__ps_func2(x_variables[n], x_variables[0], n, self.ltype, 2)
                        bb.append(b)

                g = self.__beta_func(aa, self.dtype)
                h = self.__beta_func(bb, self.dtype)

                alpha = self.__alpha_func(x_variables, 2, self.ptype)

                y_objectives[0] = alpha[0] + h
                y_objectives[1] = alpha[1] + g
            if self.ltype == 25:
                for n in range(1, self.number_of_variables):
                    if n % 3 == 0:
                        a = self.__ps_func2(x_variables[n], x_variables[0], n, self.ltype, 1)
                        aa.append(a)
                    elif n % 3 == 1:
                        b = self.__ps_func2(x_variables[n], x_variables[n], n, self.ltype, 2)
                        bb.append(b)
                    else:
                        c = self.__ps_func2(x_variables[n], x_variables[0], n, self.ltype, 3)
                        if n % 2 == 0:
                            aa.append(c)
                        else:
                            bb.append(c)

                g = self.__beta_func(aa, self.dtype)
                h = self.__beta_func(bb, self.dtype)

                alpha = self.__alpha_func(x_variables, 2, self.ptype)

                y_objectives[0] = alpha[0] + h
                y_objectives[1] = alpha[1] + g

        if self.number_of_objectives == 3:
            if self.ltype == 31 or self.ltype == 32:

                for n in range(2, self.number_of_variables):
                    a = self.__ps_func3(x_variables[n], x_variables[0], x_variables[1], n, self.ltype)

                    if n % 3 == 0:
                        aa.append(a)
                    elif n % 3 == 1:
                        bb.append(a)
                    else:
                        cc.append(a)

                g = self.__beta_func(aa, self.dtype)
                h = self.__beta_func(bb, self.dtype)
                e = self.__beta_func(cc, self.dtype)

                alpha = self.__alpha_func(x_variables, 3, self.ptype)

                y_objectives[0] = alpha[0] + h
                y_objectives[1] = alpha[1] + g
                y_objectives[2] = alpha[2] + e

        return y_objectives

    def get_name(self):
        return 'LZ09'


class LZ09_F1(LZ09):

    def __init__(self):
        super(LZ09_F1, self).__init__(number_of_variables=10, number_of_objectives=2, number_of_constraints=0,
                                      dtype=1, ltype=21, ptype=21)

    def get_name(self):
        return 'LZ09_F1'


class LZ09_F2(LZ09):

    def __init__(self):
        super(LZ09_F2, self).__init__(number_of_variables=30, number_of_objectives=2, number_of_constraints=0,
                                      dtype=1, ltype=22, ptype=21)

    def get_name(self):
        return 'LZ09_F2'


class LZ09_F3(LZ09):

    def __init__(self):
        super(LZ09_F3, self).__init__(number_of_variables=30, number_of_objectives=2, number_of_constraints=0,
                                      dtype=1, ltype=23, ptype=21)

    def get_name(self):
        return 'LZ09_F3'


class LZ09_F4(LZ09):

    def __init__(self):
        super(LZ09_F4, self).__init__(number_of_variables=30, number_of_objectives=2, number_of_constraints=0,
                                      dtype=1, ltype=24, ptype=21)

    def get_name(self):
        return 'LZ09_F4'


class LZ09_F5(LZ09):

    def __init__(self):
        super(LZ09_F5, self).__init__(number_of_variables=30, number_of_objectives=2, number_of_constraints=0,
                                      dtype=1, ltype=26, ptype=21)

    def get_name(self):
        return 'LZ09_F5'


class LZ09_F6(LZ09):

    def __init__(self):
        super(LZ09_F6, self).__init__(number_of_variables=10, number_of_objectives=3, number_of_constraints=0,
                                      dtype=1, ltype=32, ptype=31)

    def get_name(self):
        return 'LZ09_F6'


class LZ09_F7(LZ09):

    def __init__(self):
        super(LZ09_F7, self).__init__(number_of_variables=10, number_of_objectives=2, number_of_constraints=0,
                                      dtype=3, ltype=21, ptype=21)

    def get_name(self):
        return 'LZ09_F7'


class LZ09_F8(LZ09):

    def __init__(self):
        super(LZ09_F8, self).__init__(number_of_variables=10, number_of_objectives=2, number_of_constraints=0,
                                      dtype=4, ltype=21, ptype=21)

    def get_name(self):
        return 'LZ09_F8'


class LZ09_F9(LZ09):

    def __init__(self):
        super(LZ09_F9, self).__init__(number_of_variables=30, number_of_objectives=2, number_of_constraints=0,
                                      dtype=1, ltype=22, ptype=22)

    def get_name(self):
        return 'LZ09_F9'