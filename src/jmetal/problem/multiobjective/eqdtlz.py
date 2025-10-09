import math

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class Eq1_DTLZ1(FloatProblem):
    def __init__(self, number_of_variables: int = 7):

        # For Eq1_DTLZ1, the number of variables is 7 (n=7)
        # and the number of objectives is 3 (M=3) by default.

        super().__init__()

        # Problem parameters
        self._number_of_variables = number_of_variables
        self._number_of_objectives = 3  # M = 3 by default
        self._number_of_constraints = 1  # One constraint (p=1)

        # Decision variable bounds: [0,1]^n
        self.lower_bound = [0.0] * number_of_variables
        self.upper_bound = [1.0] * number_of_variables

        self.obj_directions = [self.MINIMIZE] * self._number_of_objectives
        self.obj_labels = ['f1', 'f2', 'f3']

        # Constraint parameters
        self.r = 0.4  # Radius of constraint circle
        self.epsilon = 1e-4  # Tolerance to allow numerical stability

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        # Evaluates a given solution by computing its objective values (F) and
        # the constraint violation value (H).

        x = solution.variables  # Decision variables
        M = self._number_of_objectives
        n = self._number_of_variables
        k = n - M + 1  # Number of distance-related variables

        g = 0.0
        for i in range(n - k, n):
            term = x[i] - 0.5
            g += term ** 2 - math.cos(20.0 * math.pi * term)
        g = 100 * (k + g)

        # Compute objective vector F (Pareto Front)
        f = [0.5 * (1 + g)] * M
        for i in range(M):
            for j in range(M - i - 1):
                f[i] *= x[j]
            if i > 0:
                f[i] *= 1 - x[M - i - 1]

        for i in range(M):
            solution.objectives[i] = f[i]

        # Compute constraint value H
        c = [0.5] * (M - 1)
        xx = [(x[i] - c[i]) for i in range(M - 1)]
        norm_squared = sum(xi ** 2 for xi in xx)
        h = abs(norm_squared - self.r ** 2) - self.epsilon  # Constraint violation
        solution.constraints = [h]  # Only one constraint

        return solution

    def evaluate_array(self, X):
        # Evaluates a 2D array of solutions (each row is a solution vector).

        F = []  # List to store objective vectors
        H = []  # List to store constraint violations

        for x in X:
            M = self._number_of_objectives
            n = self._number_of_variables
            k = n - M + 1

            g = 0.0
            for i in range(n - k, n):
                term = x[i] - 0.5
                g += term ** 2 - math.cos(20.0 * math.pi * term)
            g = 100 * (k + g)

            f = [0.5 * (1 + g)] * M
            for i in range(M):
                for j in range(M - i - 1):
                    f[i] *= x[j]
                if i > 0:
                    f[i] *= 1 - x[M - i - 1]

            # Compute constraint value H
            c = [0.5] * (M - 1)
            xx = [(x[i] - c[i]) for i in range(M - 1)]
            norm_squared = sum(xi ** 2 for xi in xx)
            h = abs(norm_squared - self.r ** 2) - self.epsilon  # Violation

            F.append(f)
            H.append([h])

        return F, H

    def name(self) -> str:
        return 'Eq1_DTLZ1'

    # Accessor methods
    def number_of_variables(self):
        return self._number_of_variables

    def number_of_objectives(self):
        return self._number_of_objectives

    def number_of_constraints(self):
        return self._number_of_constraints


class Eq1_IDTLZ1(FloatProblem):
    def __init__(self, number_of_variables: int = 7):

        # For Eq1_IDTLZ1, the number of variables is 7 (n=7)
        # and the number of objectives is 3 (M=3) by default.

        super().__init__()

        # Problem parameters
        self._number_of_variables = number_of_variables
        self._number_of_objectives = 3  # M = 3 by default
        self._number_of_constraints = 1  # One constraint (p=1)

        # Decision variable bounds: [0,1]^n
        self.lower_bound = [0.0] * number_of_variables
        self.upper_bound = [1.0] * number_of_variables

        self.obj_directions = [self.MINIMIZE] * self._number_of_objectives
        self.obj_labels = ['f1', 'f2', 'f3']

        # Constraint parameters
        self.r = 0.4          # Radius of constraint circle
        self.epsilon = 1e-4   # Tolerance to allow numerical stability

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        # Evaluates a given solution by computing its objective values (F) and
        # the constraint violation value (H).

        x = solution.variables
        M = self._number_of_objectives
        n = self._number_of_variables
        k = n - M + 1

        g = 0.0
        for i in range(n - k, n):
            term = x[i] - 0.5
            g += term ** 2 - math.cos(20.0 * math.pi * term)
        g = 100 * (k + g)

        # Compute objective vector F (Inverted DTLZ1)
        f = [0.5 * (1 + g)] * M
        for i in range(M):
            prod_term = 1.0
            for j in range(M - i - 1):
                prod_term *= x[j]
            if i > 0:
                prod_term *= (1 - x[M - i - 1])
            f[i] *= (1 - prod_term)

        for i in range(M):
            solution.objectives[i] = f[i]

        # Compute constraint value H
        c = [0.5] * (M - 1)
        xx = [(x[i] - c[i]) for i in range(M - 1)]
        norm_squared = sum(xi ** 2 for xi in xx)
        h = abs(norm_squared - self.r ** 2) - self.epsilon
        solution.constraints = [h]

        return solution

    def evaluate_array(self, X):
        # Evaluates a 2D array of solutions (each row is a solution vector).

        F = []  # List to store objective vectors
        H = []  # List to store constraint violations

        for x in X:
            M = self._number_of_objectives
            n = self._number_of_variables
            k = n - M + 1

            g = 0.0
            for i in range(n - k, n):
                term = x[i] - 0.5
                g += term ** 2 - math.cos(20.0 * math.pi * term)
            g = 100 * (k + g)

            f = [0.5 * (1 + g)] * M
            for i in range(M):
                prod_term = 1.0
                for j in range(M - i - 1):
                    prod_term *= x[j]
                if i > 0:
                    prod_term *= (1 - x[M - i - 1])
                f[i] *= (1 - prod_term)

            # Compute constraint value
            c = [0.5] * (M - 1)
            xx = [(x[i] - c[i]) for i in range(M - 1)]
            norm_squared = sum(xi ** 2 for xi in xx)
            h = abs(norm_squared - self.r ** 2) - self.epsilon

            F.append(f)
            H.append([h])

        return F, H

    def name(self) -> str:
        return 'Eq1_IDTLZ1'

    def number_of_variables(self):
        return self._number_of_variables

    def number_of_objectives(self):
        return self._number_of_objectives

    def number_of_constraints(self):
        return self._number_of_constraints


class Eq1_DTLZ2(FloatProblem):
    def __init__(self, number_of_variables: int = 12):

        # For Eq1_DTLZ2, the number of variables is 12 (n=12)
        # and the number of objectives is 3 (M=3) by default.

        super().__init__()

        # Problem parameters
        self._number_of_variables = number_of_variables
        self._number_of_objectives = 3  # M = 3 by default
        self._number_of_constraints = 1  # One constraint (p=1)

        # Decision variable bounds: [0,1]^n
        self.lower_bound = [0.0] * number_of_variables
        self.upper_bound = [1.0] * number_of_variables

        self.obj_directions = [self.MINIMIZE] * self._number_of_objectives
        self.obj_labels = ['f1', 'f2', 'f3']

        # Constraint parameters
        self.r = 0.4          # Radius of constraint circle
        self.epsilon = 1e-4   # Tolerance to allow numerical stability

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        # Evaluates a given solution by computing its objective values (F) and
        # the constraint violation value (H).

        x = solution.variables
        M = self._number_of_objectives
        n = self._number_of_variables
        k = n - M + 1

        g = sum((x[i] - 0.5) ** 2 for i in range(n - k, n))

        # Compute objective vector F (Pareto Front)
        f = [1.0] * M
        for i in range(M):
            product = 1 + g
            for j in range(M - i - 1):
                product *= math.cos(x[j] * math.pi / 2)
            if i > 0:
                product *= math.sin(x[M - i - 1] * math.pi / 2)
            f[i] = product

        for i in range(M):
            solution.objectives[i] = f[i]

        # Compute constraint value H
        c = [0.5] * (M - 1)
        xx = [(x[i] - c[i]) for i in range(M - 1)]
        norm_squared = sum(xi ** 2 for xi in xx)
        h = abs(norm_squared - self.r ** 2) - self.epsilon

        solution.constraints = [h]  # Only one constraint

        return solution

    def evaluate_array(self, X):
        # Evaluates a 2D array of solutions (each row is a solution vector).

        F = []  # List to store objective vectors
        H = []  # List to store constraint violations

        for x in X:
            M = self._number_of_objectives
            n = self._number_of_variables
            k = n - M + 1

            g = sum((x[i] - 0.5) ** 2 for i in range(n - k, n))

            f = [1.0] * M
            for i in range(M):
                product = 1 + g
                for j in range(M - i - 1):
                    product *= math.cos(x[j] * math.pi / 2)
                if i > 0:
                    product *= math.sin(x[M - i - 1] * math.pi / 2)
                f[i] = product

            # Compute constraint value H
            c = [0.5] * (M - 1)
            xx = [(x[i] - c[i]) for i in range(M - 1)]
            norm_squared = sum(xi ** 2 for xi in xx)
            h = abs(norm_squared - self.r ** 2) - self.epsilon

            F.append(f)
            H.append([h])

        return F, H

    def name(self) -> str:
        return 'Eq1_DTLZ2'

    def number_of_variables(self):
        return self._number_of_variables

    def number_of_objectives(self):
        return self._number_of_objectives

    def number_of_constraints(self):
        return self._number_of_constraints


class Eq1_IDTLZ2(FloatProblem):
    def __init__(self, number_of_variables: int = 12):

        # For Eq1_IDTLZ2, the number of variables is 12 (n=12)
        # and the number of objectives is 3 (M=3) by default.

        super().__init__()

        # Problem parameters
        self._number_of_variables = number_of_variables
        self._number_of_objectives = 3  # M = 3 by default
        self._number_of_constraints = 1  # One constraint (p=1)

        # Decision variable bounds: [0,1]^n
        self.lower_bound = [0.0] * number_of_variables
        self.upper_bound = [1.0] * number_of_variables

        self.obj_directions = [self.MINIMIZE] * self._number_of_objectives
        self.obj_labels = ['f1', 'f2', 'f3']

        # Constraint parameters
        self.r = 0.4  # Radius of constraint circle
        self.epsilon = 1e-4  # Tolerance to allow numerical stability

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        x = solution.variables
        M = self._number_of_objectives
        n = self._number_of_variables
        k = n - M + 1

        # Compute g(x)
        g = 0.0
        for i in range(n - k, n):
            diff = x[i] - 0.5
            g += diff * diff

        # Compute DTLZ2 objective values
        f_dtlz = [1.0 + g] * M
        for i in range(M):
            for j in range(M - i - 1):
                f_dtlz[i] *= math.cos(x[j] * math.pi / 2)
            if i > 0:
                f_dtlz[i] *= math.sin(x[M - i - 1] * math.pi / 2)

        # Inverted Pareto Front
        f = [(1 + g) - f_dtlz[i] for i in range(M)]

        for i in range(M):
            solution.objectives[i] = f[i]

        # Compute constraint value H
        c = [0.5] * (M - 1)
        xx = [(x[i] - c[i]) for i in range(M - 1)]
        norm_sq = sum(xi ** 2 for xi in xx)
        h = abs(norm_sq - self.r ** 2) - self.epsilon

        solution.constraints = [h]

        return solution

    def evaluate_array(self, X):
        F = []
        H = []

        for x in X:
            M = self._number_of_objectives
            n = self._number_of_variables
            k = n - M + 1

            g = 0.0
            for i in range(n - k, n):
                diff = x[i] - 0.5
                g += diff * diff

            f_dtlz = [1.0 + g] * M
            for i in range(M):
                for j in range(M - i - 1):
                    f_dtlz[i] *= math.cos(x[j] * math.pi / 2)
                if i > 0:
                    f_dtlz[i] *= math.sin(x[M - i - 1] * math.pi / 2)

            f = [(1 + g) - f_dtlz[i] for i in range(M)]

            c = [0.5] * (M - 1)
            xx = [(x[i] - c[i]) for i in range(M - 1)]
            norm_sq = sum(xi ** 2 for xi in xx)
            h = abs(norm_sq - self.r ** 2) - self.epsilon

            F.append(f)
            H.append([h])

        return F, H

    def name(self) -> str:
        return 'Eq1_IDTLZ2'

    def number_of_variables(self):
        return self._number_of_variables

    def number_of_objectives(self):
        return self._number_of_objectives

    def number_of_constraints(self):
        return self._number_of_constraints


class Eq2_DTLZ1(FloatProblem):
    def __init__(self, number_of_variables: int = 8):

        # For Eq2_DTLZ1, the number of variables is 8 (n=8)
        # and the number of objectives is 4 (M=4) by default.

        super().__init__()

        # Problem parameters
        self._number_of_variables = number_of_variables
        self._number_of_objectives = 4  # M = 4 by default
        self._number_of_constraints = 2  # Two constraints (p=2)

        # Decision variable bounds: [0,1]^n
        self.lower_bound = [0.0] * number_of_variables
        self.upper_bound = [1.0] * number_of_variables

        self.obj_directions = [self.MINIMIZE] * self._number_of_objectives
        self.obj_labels = ['f1', 'f2', 'f3', 'f4']

        # Constraint parameters
        self.r = 0.5  # Radius of constraint circles
        self.epsilon = 1e-2  # Tolerance to allow numerical stability

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        # Evaluates a given solution by computing its objective values (F) and
        # the constraint violation values (H).

        x = solution.variables
        M = self._number_of_objectives
        n = self._number_of_variables
        k = n - M + 1

        g = 0.0
        for i in range(n - k, n):
            term = x[i] - 0.5
            g += term ** 2 - math.cos(20.0 * math.pi * term)
        g = 100 * (k + g)

        # Compute objective vector F (Pareto Front)
        f = [0.5 * (1 + g)] * M
        for i in range(M):
            for j in range(M - i - 1):
                f[i] *= x[j]
            if i > 0:
                f[i] *= 1 - x[M - i - 1]

        for i in range(M):
            solution.objectives[i] = f[i]

        # Compute constraint values H
        c = [0.5] * (M - 1)
        xx = [(x[i] - c[i]) for i in range(M - 1)]
        norm1 = sum(xi ** 2 for xi in xx)
        h1 = abs(norm1 - self.r ** 2) - self.epsilon

        yy = list(xx)
        yy[-1] -= self.r
        norm2 = sum(yi ** 2 for yi in yy)
        h2 = abs(norm2 - self.r ** 2) - self.epsilon

        solution.constraints = [h1, h2]

        return solution

    def evaluate_array(self, X):
        # Evaluates a 2D array of solutions (each row is a solution vector).

        F = []  # List to store objective vectors
        H = []  # List to store constraint violations

        for x in X:
            M = self._number_of_objectives
            n = self._number_of_variables
            k = n - M + 1

            g = 0.0
            for i in range(n - k, n):
                term = x[i] - 0.5
                g += term ** 2 - math.cos(20.0 * math.pi * term)
            g = 100 * (k + g)

            f = [0.5 * (1 + g)] * M
            for i in range(M):
                for j in range(M - i - 1):
                    f[i] *= x[j]
                if i > 0:
                    f[i] *= 1 - x[M - i - 1]

            # Compute constraint values
            c = [0.5] * (M - 1)
            xx = [(x[i] - c[i]) for i in range(M - 1)]
            norm1 = sum(xi ** 2 for xi in xx)
            h1 = abs(norm1 - self.r ** 2) - self.epsilon

            yy = list(xx)
            yy[-1] -= self.r
            norm2 = sum(yi ** 2 for yi in yy)
            h2 = abs(norm2 - self.r ** 2) - self.epsilon

            F.append(f)
            H.append([h1, h2])

        return F, H

    def name(self) -> str:
        return 'Eq2_DTLZ1'

    # Accessor methods
    def number_of_variables(self):
        return self._number_of_variables

    def number_of_objectives(self):
        return self._number_of_objectives

    def number_of_constraints(self):
        return self._number_of_constraints


class Eq2_IDTLZ1(FloatProblem):
    def __init__(self, number_of_variables: int = 8):

        # For Eq2_IDTLZ1, the number of variables is 8 (n=8)
        # and the number of objectives is 4 (M=4) by default.

        super().__init__()

        # Problem parameters
        self._number_of_variables = number_of_variables
        self._number_of_objectives = 4  # M = 4 by default
        self._number_of_constraints = 2  # Two constraints (p=2)

        # Decision variable bounds: [0,1]^n
        self.lower_bound = [0.0] * number_of_variables
        self.upper_bound = [1.0] * number_of_variables

        self.obj_directions = [self.MINIMIZE] * self._number_of_objectives
        self.obj_labels = ['f1', 'f2', 'f3', 'f4']

        # Constraint parameters
        self.r = 0.5  # Radius of constraint circles
        self.epsilon = 1e-2  # Tolerance to allow numerical stability

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        # Evaluates a given solution by computing its objective values (F) and
        # the constraint violation values (H).

        x = solution.variables
        M = self._number_of_objectives
        n = self._number_of_variables
        k = n - M + 1

        g = 0.0
        for i in range(n - k, n):
            term = x[i] - 0.5
            g += term ** 2 - math.cos(20.0 * math.pi * term)
        g = 100 * (k + g)

        # Compute objective vector F (Inverted Pareto Front)
        f = [0.5 * (1 + g)] * M
        for i in range(M):
            prod_term = 1.0
            for j in range(M - i - 1):
                prod_term *= x[j]
            if i > 0:
                prod_term *= (1 - x[M - i - 1])
            f[i] *= (1 - prod_term)

        for i in range(M):
            solution.objectives[i] = f[i]

        # Compute constraint values H
        c = [0.5] * (M - 1)
        xx = [(x[i] - c[i]) for i in range(M - 1)]
        norm1 = sum(xi ** 2 for xi in xx)
        h1 = abs(norm1 - self.r ** 2) - self.epsilon

        yy = list(xx)
        yy[-1] -= self.r
        norm2 = sum(yi ** 2 for yi in yy)
        h2 = abs(norm2 - self.r ** 2) - self.epsilon

        solution.constraints = [h1, h2]

        return solution

    def evaluate_array(self, X):
        # Evaluates a 2D array of solutions (each row is a solution vector).

        F = []  # List to store objective vectors
        H = []  # List to store constraint violations

        for x in X:
            M = self._number_of_objectives
            n = self._number_of_variables
            k = n - M + 1

            g = 0.0
            for i in range(n - k, n):
                term = x[i] - 0.5
                g += term ** 2 - math.cos(20.0 * math.pi * term)
            g = 100 * (k + g)

            f = [0.5 * (1 + g)] * M
            for i in range(M):
                prod_term = 1.0
                for j in range(M - i - 1):
                    prod_term *= x[j]
                if i > 0:
                    prod_term *= (1 - x[M - i - 1])
                f[i] *= (1 - prod_term)

            # Compute constraint values
            c = [0.5] * (M - 1)
            xx = [(x[i] - c[i]) for i in range(M - 1)]
            norm1 = sum(xi ** 2 for xi in xx)
            h1 = abs(norm1 - self.r ** 2) - self.epsilon

            yy = list(xx)
            yy[-1] -= self.r
            norm2 = sum(yi ** 2 for yi in yy)
            h2 = abs(norm2 - self.r ** 2) - self.epsilon

            F.append(f)
            H.append([h1, h2])

        return F, H

    def name(self) -> str:
        return 'Eq2_IDTLZ1'

    def number_of_variables(self):
        return self._number_of_variables

    def number_of_objectives(self):
        return self._number_of_objectives

    def number_of_constraints(self):
        return self._number_of_constraints


class Eq2_DTLZ2(FloatProblem):
    def __init__(self, number_of_variables: int = 13):

        # For Eq2_DTLZ2, the number of variables is 13 (n=13)
        # and the number of objectives is 4 (M=4) by default.

        super().__init__()

        # Problem parameters
        self._number_of_variables = number_of_variables
        self._number_of_objectives = 4  # M = 4 by default
        self._number_of_constraints = 2  # Two constraints (p=2)

        # Decision variable bounds: [0,1]^n
        self.lower_bound = [0.0] * number_of_variables
        self.upper_bound = [1.0] * number_of_variables

        self.obj_directions = [self.MINIMIZE] * self._number_of_objectives
        self.obj_labels = ['f1', 'f2', 'f3', 'f4']

        # Constraint parameters
        self.r = 0.5         # Radius of constraint circles
        self.epsilon = 1e-2  # Tolerance to allow numerical stability

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        # Evaluates a given solution by computing its objective values (F) and
        # the constraint violation values (H).

        x = solution.variables
        M = self._number_of_objectives
        n = self._number_of_variables
        k = n - M + 1

        g = sum((x[i] - 0.5) ** 2 for i in range(n - k, n))

        # Compute objective vector F (Pareto Front)
        f = [1.0] * M
        for i in range(M):
            product = 1 + g
            for j in range(M - i - 1):
                product *= math.cos(x[j] * math.pi / 2)
            if i > 0:
                product *= math.sin(x[M - i - 1] * math.pi / 2)
            f[i] = product

        for i in range(M):
            solution.objectives[i] = f[i]

        # Compute constraint values H
        c = [0.5] * (M - 1)
        xx = [(x[i] - c[i]) for i in range(M - 1)]
        norm1 = sum(xi ** 2 for xi in xx)
        h1 = abs(norm1 - self.r ** 2) - self.epsilon

        yy = list(xx)
        yy[-1] -= self.r
        norm2 = sum(yi ** 2 for yi in yy)
        h2 = abs(norm2 - self.r ** 2) - self.epsilon

        solution.constraints = [h1, h2]

        return solution

    def evaluate_array(self, X):
        # Evaluates a 2D array of solutions (each row is a solution vector).

        F = []  # List to store objective vectors
        H = []  # List to store constraint violations

        for x in X:
            M = self._number_of_objectives
            n = self._number_of_variables
            k = n - M + 1

            g = sum((x[i] - 0.5) ** 2 for i in range(n - k, n))

            f = [1.0] * M
            for i in range(M):
                product = 1 + g
                for j in range(M - i - 1):
                    product *= math.cos(x[j] * math.pi / 2)
                if i > 0:
                    product *= math.sin(x[M - i - 1] * math.pi / 2)
                f[i] = product

            # Compute constraint values
            c = [0.5] * (M - 1)
            xx = [(x[i] - c[i]) for i in range(M - 1)]
            norm1 = sum(xi ** 2 for xi in xx)
            h1 = abs(norm1 - self.r ** 2) - self.epsilon

            yy = list(xx)
            yy[-1] -= self.r
            norm2 = sum(yi ** 2 for yi in yy)
            h2 = abs(norm2 - self.r ** 2) - self.epsilon

            F.append(f)
            H.append([h1, h2])

        return F, H

    def name(self) -> str:
        return 'Eq2_DTLZ2'

    def number_of_variables(self):
        return self._number_of_variables

    def number_of_objectives(self):
        return self._number_of_objectives

    def number_of_constraints(self):
        return self._number_of_constraints


class Eq2_IDTLZ2(FloatProblem):
    def __init__(self, number_of_variables: int = 13):

        # For Eq2_IDTLZ2, the number of variables is 13 (n=13)
        # and the number of objectives is 4 (M=4) by default.

        super().__init__()

        # Problem parameters
        self._number_of_variables = number_of_variables
        self._number_of_objectives = 4  # M = 4 by default
        self._number_of_constraints = 2  # Two constraints (p=2)

        # Decision variable bounds: [0,1]^n
        self.lower_bound = [0.0] * number_of_variables
        self.upper_bound = [1.0] * number_of_variables

        self.obj_directions = [self.MINIMIZE] * self._number_of_objectives
        self.obj_labels = ['f1', 'f2', 'f3', 'f4']

        # Constraint parameters
        self.r = 0.5  # Radius of constraint circles
        self.epsilon = 1e-2  # Tolerance to allow numerical stability

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        x = solution.variables
        M = self._number_of_objectives
        n = self._number_of_variables
        k = n - M + 1

        # Compute g(x)
        g = 0.0
        for i in range(n - k, n):
            diff = x[i] - 0.5
            g += diff * diff

        # Compute DTLZ2 objectives (F_dtlz)
        f_dtlz = [1.0 + g] * M
        for i in range(M):
            for j in range(M - i - 1):
                f_dtlz[i] *= math.cos(x[j] * math.pi / 2)
            if i > 0:
                f_dtlz[i] *= math.sin(x[M - i - 1] * math.pi / 2)

        # Inverted Pareto Front
        f = [(1 + g) - f_dtlz[i] for i in range(M)]

        for i in range(M):
            solution.objectives[i] = f[i]

        # Compute constraints
        c = [0.5] * (M - 1)
        xx = [(x[i] - c[i]) for i in range(M - 1)]
        norm1 = sum(xi ** 2 for xi in xx)
        h1 = abs(norm1 - self.r ** 2) - self.epsilon

        yy = list(xx)
        yy[-1] -= self.r
        norm2 = sum(yi ** 2 for yi in yy)
        h2 = abs(norm2 - self.r ** 2) - self.epsilon

        solution.constraints = [h1, h2]

        return solution

    def evaluate_array(self, X):
        F = []
        H = []

        for x in X:
            M = self._number_of_objectives
            n = self._number_of_variables
            k = n - M + 1

            g = 0.0
            for i in range(n - k, n):
                diff = x[i] - 0.5
                g += diff * diff

            f_dtlz = [1.0 + g] * M
            for i in range(M):
                for j in range(M - i - 1):
                    f_dtlz[i] *= math.cos(x[j] * math.pi / 2)
                if i > 0:
                    f_dtlz[i] *= math.sin(x[M - i - 1] * math.pi / 2)

            f = [(1 + g) - f_dtlz[i] for i in range(M)]

            c = [0.5] * (M - 1)
            xx = [(x[i] - c[i]) for i in range(M - 1)]
            norm1 = sum(xi ** 2 for xi in xx)
            h1 = abs(norm1 - self.r ** 2) - self.epsilon

            yy = list(xx)
            yy[-1] -= self.r
            norm2 = sum(yi ** 2 for yi in yy)
            h2 = abs(norm2 - self.r ** 2) - self.epsilon

            F.append(f)
            H.append([h1, h2])

        return F, H

    def name(self) -> str:
        return 'Eq2_IDTLZ2'

    def number_of_variables(self):
        return self._number_of_variables

    def number_of_objectives(self):
        return self._number_of_objectives

    def number_of_constraints(self):
        return self._number_of_constraints


