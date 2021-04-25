# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 15:04:34 2020

@author: María Fdez Hijano
"""

""" jmetal imports"""
from jmetal.algorithm.multiobjective.nsgaii import DynamicNSGAII
from jmetal.operator import PermutationSwapMutation, PMXCrossover, BinaryTournamentSelection
from jmetal.util.observer import WriteFrontToFileObserver #, PlotFrontToFileObserver
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.comparator  import RankingAndCrowdingDistanceComparator

from jmetal.problem.multiobjective.cvrp import DynamicCVRP
from jmetal.util.observer import WriteFrontVariablesToFileObserver, PlotParetoFrontToFileObserver
from jmetal.util.observable import FileObservable
from jmetal.util.reader import Reader

if __name__ == '__main__':
    
    
    #INITIAL READER
    num_vehicles = 5
    filename = 'resources/CVRP_instances/A-n32-k5-dummy_points.vrp'
    filename_time = 'resources/CVRP_instances/A-n32-k5-time-dummy_points.vrp'
    
    reader = Reader(filename, num_vehicles)
    reader_time = Reader(filename_time, num_vehicles)
    distance_matrix, distance_to_warehouse = reader.read_from_file()
    demand_section                         = reader.read_demand_from_file()   
    dimension, vehicle_capacity            = reader.read_dimension_and_capacity_from_file()           
    cost_matrix, cost_to_warehouse         = reader_time.read_from_file()
    
    #PROBLEM
    problem = DynamicCVRP(num_vehicles, 
                          distance_matrix, 
                          distance_to_warehouse, 
                          cost_matrix, cost_to_warehouse, 
                          demand_section, 
                          dimension, 
                          vehicle_capacity)       
  
    #OBSERVABLE
    file_observable = FileObservable(cost_matrix, cost_to_warehouse, filename_time, num_vehicles)
    file_observable.observable.register(problem)
    file_observable.start()     
    
    #ALGORITHM
    max_evaluations = 25000
    algorithm = DynamicNSGAII(
        problem=problem,
        population_size=dimension,
        offspring_population_size=dimension,
        mutation=PermutationSwapMutation(probability=0.2),
        crossover=PMXCrossover(probability=0.9),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator())    
    )
    
    #RUNNER
    algorithm.observable.register(observer=WriteFrontToFileObserver('dynamic_front'))
    algorithm.observable.register(observer=WriteFrontVariablesToFileObserver('dynamic_var_front'))
    algorithm.observable.register(observer=PlotParetoFrontToFileObserver('dynamic_front_vis'))
    algorithm.run() 

    
