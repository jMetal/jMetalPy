from jmetal.algorithm import EvolutionStrategy
from jmetal.operator import Polynomial
from jmetal.problem import Sphere

if __name__ == '__main__':
    problem = Sphere(number_of_variables=10)

    algorithm = EvolutionStrategy(
        problem=problem,
        mu=10,
        lambda_=10,
        max_evaluations=50000,
        mutation=Polynomial(probability=1.0 / problem.number_of_variables)
    )

    algorithm.execute()
    result = algorithm.get_result()

    print('Algorithm: {}'.format(algorithm.get_name()))
    print('Problem: {}'.format(problem.get_name()))
    print('Solution: {}'.format(result.variables))
    print('Fitness: {}'.format(result.objectives[0]))
    print('Computing time: {}'.format(algorithm.total_computing_time))
