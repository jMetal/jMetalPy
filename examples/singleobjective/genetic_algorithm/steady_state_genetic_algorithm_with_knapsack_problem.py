from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator.crossover import SPXCrossover
from jmetal.operator.mutation import BitFlipMutation
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.problem.singleobjective.knapsack import Knapsack
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == "__main__":
    problem = Knapsack(
        from_file=True, filename="resources/Knapsack_instances/KnapsackInstance_50_0_0.kp"
    )

    algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=100,
        offspring_population_size=1,
        mutation=BitFlipMutation(probability=0.1),
        crossover=SPXCrossover(probability=0.8),
        selection=BinaryTournamentSelection(),
        termination_criterion=StoppingByEvaluations(max_evaluations=25000),
    )

    algorithm.run()
    subset = algorithm.result()

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Solution: {subset.variables}")
    print(f"Fitness: {-subset.objectives[0]}")
    print(f"Computing time: {algorithm.total_computing_time}")
    print(f"Problem Maximum Capacity: {problem.capacity}")
