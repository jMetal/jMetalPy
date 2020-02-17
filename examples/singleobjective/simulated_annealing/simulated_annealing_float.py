from jmetal.algorithm.singleobjective.simulated_annealing import SimulatedAnnealing
from jmetal.operator import PolynomialMutation
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = Rastrigin(10)

    max_evaluations = 250000

    algorithm = SimulatedAnnealing(
        problem=problem,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20.0),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )

    algorithm.run()
    result = algorithm.get_result()

    # Save results to file
    print_function_values_to_file(result, 'FUN.'+ algorithm.get_name() + "." + problem.get_name())
    print_variables_to_file(result, 'VAR.' + algorithm.get_name() + "." + problem.get_name())

    print('Algorithm: ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + str(result.variables[0]))
    print('Fitness:  ' + str(result.objectives[0]))
    print('Computing time: ' + str(algorithm.total_computing_time))
