# -*- coding: utf-8 -*-
"""

@author: María Fdez Hijano

"""
 
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator import PMXCrossover, PermutationSwapMutation
from jmetal.problem.multiobjective.cvrp import CVRP
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.lab.visualization import Plot



num_vehicles = 10
dimension = 88
problem = CVRP('resources/CVRP_instances//A-n80-k10-dummy_points.vrp', 'resources/CVRP_instances//A-n80-k10-time-dummy_points.vrp', num_vehicles)

max_evaluations = 150000

algorithm = NSGAII(
    problem=problem,
    population_size=dimension,
    offspring_population_size=dimension,
    mutation=PermutationSwapMutation(probability=0.2),
    crossover=PMXCrossover(probability=0.9),
    termination_criterion = StoppingByEvaluations(max_evaluations=max_evaluations)
)

algorithm.run()  

front = get_non_dominated_solutions(algorithm.get_result())


# save to files


print_function_values_to_file(front, 'output/tmp/FUN.'+ algorithm.get_name()+"-"+problem.get_name())
print_variables_to_file(front, 'output/tmp/VAR.' + algorithm.get_name()+"-"+problem.get_name())
plot_front = Plot(title='Pareto front approximation', axis_labels=['distance cost', 'time cost'])
plot_front.plot(front, label='NSGAII-CVRP (25000 evals)', filename='output/tmp/NSGAII-CVRP', format='png')

print('Algorithm (continuous problem): ' + algorithm.get_name())
print('Problem: ' + problem.get_name())
print('Computing time: ' + str(algorithm.total_computing_time))
