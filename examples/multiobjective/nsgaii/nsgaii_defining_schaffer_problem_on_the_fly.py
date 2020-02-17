from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.problem import OnTheFlyFloatProblem
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.solution import get_non_dominated_solutions, read_solutions, print_function_values_to_file, \
    print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

"""  
Program to  configure and run the NSGA-II algorithm configured with standard settings.
"""

if __name__ == '__main__':
    # Defining problem Schaffer on the fly
    def f1(x: [float]):
        return x[0] * x[0]

    def f2(x: [float]):
        return (x[0] - 2) * (x[0] - 2)

    problem = OnTheFlyFloatProblem()
    problem \
        .set_name('Schaffer') \
        .add_variable(-10000.0, 10000.0) \
        .add_function(f1) \
        .add_function(f2)

    problem.reference_front = read_solutions(filename='resources/reference_front/ZDT1.pf')

    max_evaluations = 25000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )

    algorithm.run()
    front = get_non_dominated_solutions(algorithm.get_result())

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.label)
    print_variables_to_file(front, 'VAR.'+ algorithm.label)

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))

