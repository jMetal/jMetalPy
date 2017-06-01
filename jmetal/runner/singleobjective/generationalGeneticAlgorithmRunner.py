from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.core.solution import BinarySolution, FloatSolution
from jmetal.operator.crossover import SinglePoint, SBX
from jmetal.operator.mutation import BitFlip, Polynomial
from jmetal.operator.selection import BinaryTournament
from jmetal.problem.singleobjectiveproblem import OneMax, Sphere


def main():
    binary_example()
    float_example()
    run_as_a_thread_example()

def binary_example() -> None:
    bits = 512
    problem = OneMax(bits)
    algorithm = GenerationalGeneticAlgorithm[BinarySolution, BinarySolution](
        problem,
        population_size = 100,
        max_evaluations = 25000,
        mutation = BitFlip(1.0/bits),
        crossover = SinglePoint(0.9),
        selection = BinaryTournament())

    algorithm.run()
    result = algorithm.get_result()
    print("Algorithm: " + algorithm.get_name())
    print("Problem: " + problem.get_name())
    print("Solution: " + str(result.variables[0]))
    print("Fitness:  " + str(result.objectives[0]))


def run_as_a_thread_example() -> None:
    variables = 10
    problem = Sphere(variables)
    algorithm = GenerationalGeneticAlgorithm[FloatSolution, FloatSolution](
        problem,
        population_size = 100,
        max_evaluations = 25000,
        mutation = Polynomial(1.0/variables, distribution_index=20),
        crossover = SBX(1.0, distribution_index=20),
        selection = BinaryTournament())

    algorithm.start()
    print("Algorithm (running as a thread): " + algorithm.get_name())
    print("Problem: " + problem.get_name())

    algorithm.join()
    result = algorithm.get_result()
    print("Solution: " + str(result.variables))
    print("Fitness:  " + str(result.objectives[0]))


def float_example() -> None:
    variables = 10
    problem = Sphere(variables)
    algorithm = GenerationalGeneticAlgorithm[FloatSolution, FloatSolution](
        problem,
        population_size = 100,
        max_evaluations = 25000,
        mutation = Polynomial(1.0/variables, distribution_index=20),
        crossover = SBX(1.0, distribution_index=20),
        selection = BinaryTournament())

    algorithm.run()
    result = algorithm.get_result()
    print("Algorithm: " + algorithm.get_name())
    print("Problem: " + problem.get_name())
    print("Solution: " + str(result.variables))
    print("Fitness:  " + str(result.objectives[0]))

if __name__ == '__main__':
    main()
