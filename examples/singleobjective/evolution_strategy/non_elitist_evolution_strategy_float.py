from jmetal.algorithm.singleobjective.evolutionaryalgorithm import NonElitistEvolutionStrategy
from jmetal.core.solution import FloatSolution
from jmetal.operator.mutation import Polynomial
from jmetal.problem.singleobjective.unconstrained import Sphere


def main() -> None:
    variables = 10
    problem = Sphere(variables)
    algorithm = NonElitistEvolutionStrategy[FloatSolution, FloatSolution]\
        (problem, mu=10, lambdA=10, max_evaluations= 50000, mutation=Polynomial(1.0/variables))

    algorithm.run()
    result = algorithm.get_result()
    print("Algorithm: " + algorithm.get_name())
    print("Problem: " + problem.get_name())
    print("Solution: " + str(result.variables))
    print("Fitness:  " + str(result.objectives[0]))
    print("Computing time: " + str(algorithm.total_computing_time))


if __name__ == '__main__':
    main()
