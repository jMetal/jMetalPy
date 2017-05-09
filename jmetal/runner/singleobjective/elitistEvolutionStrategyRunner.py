from jmetal.algorithm.singleobjective.evolutionaryalgorithm.elitistEvolutionStrategy import ElitistEvolutionStrategy
from jmetal.core.solution.binarySolution import BinarySolution
from jmetal.core.solution.floatSolution import FloatSolution
from jmetal.operator.mutation.bitflip import BitFlip
from jmetal.operator.mutation.polynomial import Polynomial
from jmetal.problem.singleobjective.onemax import OneMax
from jmetal.problem.singleobjective.sphere import Sphere


def main():
    binary_example()
    print()
    float_example()


def binary_example() -> None:
    bits = 512
    problem = OneMax(bits)
    algorithm = ElitistEvolutionStrategy[BinarySolution, BinarySolution]\
        (problem, mu=1, lambdA=10, max_evaluations= 25000, mutation_operator=BitFlip(1.0/bits))

    algorithm.run()
    result = algorithm.get_result()
    print("Algorithm: " + algorithm.get_name())
    print("Problem: " + problem.get_name())
    print("Solution: " + str(result.variables[0]))
    print("Fitness:  " + str(result.objectives[0]))


def float_example() -> None:
    variables = 10
    problem = Sphere(variables)
    algorithm = ElitistEvolutionStrategy[FloatSolution, FloatSolution]\
        (problem, mu=10, lambdA=10, max_evaluations= 50000, mutation_operator=Polynomial(1.0/variables))

    algorithm.run()
    result = algorithm.get_result()
    print("Algorithm: " + algorithm.get_name())
    print("Problem: " + problem.get_name())
    print("Solution: " + str(result.variables))
    print("Fitness:  " + str(result.objectives[0]))

if __name__ == '__main__':
    main()
