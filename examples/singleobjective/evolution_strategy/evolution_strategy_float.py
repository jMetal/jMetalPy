from jmetal.algorithm.singleobjective.evolution_strategy import EvolutionStrategy
from jmetal.operator import PolynomialMutation
from jmetal.problem import Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = Sphere(number_of_variables=10)

    algorithm = EvolutionStrategy(
        problem=problem,
        mu=10,
        lambda_=10,
        elitist=True,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables),
        termination_criterion=StoppingByEvaluations(max_evaluations=25000)
    )

    algorithm.run()
    result = algorithm.get_result()

    print('Algorithm: ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + str(result.variables[0]))
    print('Fitness:  ' + str(result.objectives[0]))
    print('Computing time: ' + str(algorithm.total_computing_time))
