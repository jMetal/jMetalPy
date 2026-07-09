from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator.crossover import PMXCrossover
from jmetal.operator.mutation import PermutationSwapMutation
from jmetal.problem.singleobjective.tsp import TSP
from jmetal.util.observer import PrintObjectivesObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == "__main__":
    problem = TSP(instance="kroA100.tsp")

    print("Cities: ", problem.number_of_variables())

    algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        # original mutation probability (1/n)
        mutation=PermutationSwapMutation(1.0 / problem.number_of_variables()),
        crossover=PMXCrossover(0.8),
        termination_criterion=StoppingByEvaluations(max_evaluations=100000),
    )

    algorithm.observable.register(observer=PrintObjectivesObserver(1000))

    algorithm.run()
    result = algorithm.result()

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Solution: {result.variables}")
    print(f"Fitness: {result.objectives[0]}")
    print(f"Computing time: {algorithm.total_computing_time}")
