from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator import SPXCrossover, BitFlipMutation, BinaryTournamentSelection
from jmetal.problem.singleobjective.unconstrained import SubsetSum
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    C = 300500
    W = [2902, 5235, 357, 6058, 4846, 8280, 1295, 181, 3264,
         7285, 8806, 2344, 9203, 6806, 1511, 2172, 843, 4697,
         3348, 1866, 5800, 4094, 2751, 64, 7181, 9167, 5579,
         9461, 3393, 4602, 1796, 8174, 1691, 8854, 5902, 4864,
         5488, 1129, 1111, 7597, 5406, 2134, 7280, 6465, 4084,
         8564, 2593, 9954, 4731, 1347, 8984, 5057, 3429, 7635,
         1323, 1146, 5192, 6547, 343, 7584, 3765, 8660, 9318,
         5098, 5185, 9253, 4495, 892, 5080, 5297, 9275, 7515,
         9729, 6200, 2138, 5480, 860, 8295, 8327, 9629, 4212,
         3087, 5276, 9250, 1835, 9241, 1790, 1947, 8146, 8328,
         973, 1255, 9733, 4314, 6912, 8007, 8911, 6802, 5102,
         5451, 1026, 8029, 6628, 8121, 5509, 3603, 6094, 4447,
         683, 6996, 3304, 3130, 2314, 7788, 8689, 3253, 5920,
         3660, 2489, 8153, 2822, 6132, 7684, 3032, 9949, 59,
         6669, 6334]

    problem = SubsetSum(C, W)

    algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=100,
        offspring_population_size=1,
        mutation=BitFlipMutation(probability=0.1),
        crossover=SPXCrossover(probability=0.8),
        selection=BinaryTournamentSelection(),
        termination_criterion=StoppingByEvaluations(max_evaluations=25000)
    )

    algorithm.run()
    subset = algorithm.get_result()

    print('Algorithm: {}'.format(algorithm.get_name()))
    print('Problem: {}'.format(problem.get_name()))
    print('Solution: {}'.format(subset.variables))
    print('Fitness: {}'.format(subset.objectives[0]))
    print('Computing time: {}'.format(algorithm.total_computing_time))
