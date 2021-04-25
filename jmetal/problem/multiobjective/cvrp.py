# -*- coding: utf-8 -*-
"""
Created on Mon Dec 7 19:56:17 2020

@author: María Fdez Hijano
"""

import math
import random
import re

from jmetal.core.problem import PermutationProblem, DynamicProblem
from jmetal.core.solution import PermutationSolution
from jmetal.util.observable import FileObservable
from abc import ABC

"""
.. module:: cvrp
   :platform: Unix, Windows
   :synopsis: Capacitated Vehicle Routing Problem

.. moduleauthor:: María Fdez Hijano
"""

class CVRP(PermutationProblem):
    """ Class representing TSP Problem. """

    def __init__(self, instance: str = None, instance2: str = None, num_vehicles: int = 1):
        super(CVRP, self).__init__()
        
        self.num_vehicles = num_vehicles
        """It's not necessary to use all of them"""
        
        cost_matrix, cost_to_warehouse, demand_section, dimension, vehicle_capacity = self.__read_from_file(instance2)
        distance_matrix, distance_to_warehouse, demand_section, dimension, vehicle_capacity = self.__read_from_file(instance)

        self.distance_matrix = distance_matrix
        self.cost_matrix = cost_matrix
        self.distance_to_warehouse = distance_to_warehouse
        self.cost_to_warehouse = cost_to_warehouse
        self.demand_section = demand_section
        self.vehicle_capacity = vehicle_capacity
        self.obj_directions = [self.MINIMIZE]
        self.number_of_variables = dimension
        self.number_of_objectives = 2
        self.number_of_constraints = 0

    def __read_from_file(self, filename: str):
        """
        This function reads a CVRP Problem instance from a file.

        :param filename: File which describes the instance.
        :type filename: str.
        """

        if filename is None:
            raise FileNotFoundError('Filename can not be None')

        with open(filename) as file:    
            lines = file.readlines()
            data = [line.lstrip() for line in lines if line != ""]            

            dimension = re.compile(r'[^\d]+')
            vehicle_capacity = re.compile(r'[^\d]+')         
            dummy_nodes = self.num_vehicles
            
            for item in data:
                if item.startswith('DIMENSION'):
                    dimension = int(dimension.sub('', item)) + dummy_nodes
                elif item.startswith('CAPACITY'):
                    vehicle_capacity = int(vehicle_capacity.sub('', item))                             
                elif item.startswith("NODE_COORD_SECTION"):                       
                    delivery_points_x = [-1] * (dimension - 1)
                    delivery_points_y = [-1] * (dimension - 1)
                    depot_x = 0
                    depot_y = 0        
                    for item in data:
                        if item[0].isdigit():
                            j, dp_x, dp_y = [int(x.strip()) for x in item.split(' ')]
                            if j == 1:
                                """origin x"""
                                depot_x = dp_x
                                """origin y"""
                                depot_y = dp_y 
                            else:                       
                                delivery_points_x[j-2] = dp_x
                                delivery_points_y[j-2] = dp_y
                        if item.startswith('DEMAND_SECTION'):
                            break
        
                    """ Compute distances between delivery points"""
                    distance_matrix = [[-1] * (dimension - 1) for _ in range(dimension - 1)]
                    for k in range(dimension - 1):
                        distance_matrix[k][k] = 0
                        for j in range(dimension - 1):
                            dist = math.sqrt((delivery_points_x[k] - delivery_points_x[j]) ** 2 + (delivery_points_y[k] - delivery_points_y[j]) ** 2)
                            dist = round(dist)
                            distance_matrix[k][j] = dist
                            distance_matrix[j][k] = dist
                   
                        """ Compute distances to warehouse"""
                    distance_to_warehouse =  [-1] * (dimension - 1)
                    """distance_to_warehouse[k] = 0"""
                    for j in range(dimension - 1):
                        dist = math.sqrt((depot_x - delivery_points_x[j]) ** 2 + (depot_y - delivery_points_y[j]) ** 2)
                        dist = round(dist)
                        distance_to_warehouse[j] = dist
                    break    
                
            for item in data: 
                if item.startswith('DEMAND_SECTION'):
                    demand_section = [-1] * (dimension - 1)
                    demand_section[0] = 0
                    pos = data.index(item) + 2      
                    for item in data[pos:len(data)]:                         
                        if item[0].isdigit():
                            item.replace(" \n", '\n')
                            j, demand, s = [x.strip() for x in item.split(' ')]
                            j = int(j)
                            demand_section[j-2] = int(demand)
                        if item.startswith('DEPOT_SECTION'):
                            break                                  
    
            return distance_matrix, distance_to_warehouse, demand_section, len(delivery_points_x), vehicle_capacity
        

    def evaluate(self, solution: PermutationSolution) -> PermutationSolution:
        fitness1 = 0
        fitness2 = 0
        route_demand = 0

        for i in range(self.number_of_variables - 2):
            x = solution.variables[i]
            y = solution.variables[i + 1] 
             
            if i == 0:
                fitness1 += self.distance_to_warehouse[x]    
                fitness2 += self.cost_to_warehouse[x]                
                route_demand += self.demand_section[x-1]  
                
            route_demand += self.demand_section[y-1] 
            if y > (self.number_of_variables - self.num_vehicles):              
                route_demand = 0    
            if route_demand > self.vehicle_capacity:
                fitness1 += 99999999
                fitness2 += 99999999
            fitness1 += self.distance_matrix[x][y]       
            fitness2 += self.cost_matrix[x][y]
                     
        solution.objectives[0] = fitness1
        solution.objectives[1] = fitness2

        """ ${f_{eval}(x) = f_{max} – f(x)}$, where ${f(x) = totaldistance(x) + \lambda overcapacity(x) + \mu overtime(x)}$ """      

        return solution
    
    
    def create_solution(self) -> PermutationSolution:
        new_solution = PermutationSolution(number_of_variables=self.number_of_variables,
                                           number_of_objectives=self.number_of_objectives)
     
        destination = [self.number_of_variables]
        #random.seed(1)
        new_solution.variables = random.sample(range(1,self.number_of_variables), k=self.number_of_variables - 1) + destination

        return new_solution
 
    
    @property
    def number_of_cities(self):
        return self.number_of_variables

    def get_name(self):
        return 'CVRP'


class DynamicCVRP(DynamicProblem, PermutationProblem, FileObservable, ABC):
    """ Class representing Dynamic CVRP Problem. """

    def __init__(self, 
                 num_vehicles, 
                 distance_matrix, distance_to_warehouse, 
                 cost_matrix, cost_to_warehouse,
                 demand_section,
                 dimension, 
                 vehicle_capacity):        
        super(DynamicCVRP, self).__init__()
        
        self.problem_modified = False        
        
        self.num_vehicles = num_vehicles  # It's not necessary to use all of them
        self.distance_matrix = distance_matrix
        self.distance_to_warehouse = distance_to_warehouse
        self.cost_matrix = cost_matrix
        self.cost_to_warehouse = cost_to_warehouse
        self.demand_section = demand_section
        self.vehicle_capacity = vehicle_capacity
        self.obj_directions = [self.MINIMIZE]
        self.number_of_variables = dimension
        self.number_of_objectives = 2
        self.number_of_constraints = 0       
        
    def update(self, *args, **kwargs):        
        if kwargs['COST_MATRIX'] != self.cost_matrix or kwargs['COST_TO_WAREHOUSE'] != self.cost_to_warehouse:
            self.cost_matrix = kwargs['COST_MATRIX']
            self.cost_to_warehouse = kwargs['COST_TO_WAREHOUSE']    
            print('\nOK - CHANGES')
            self.problem_modified = True          

    def the_problem_has_changed(self) -> bool:
        return self.problem_modified

    def clear_changed(self) -> None:
        self.problem_modified = False

    def evaluate(self, solution: PermutationSolution) -> PermutationSolution:
        fitness1 = 0
        fitness2 = 0
        route_demand = 0

        for i in range(self.number_of_variables - 2):
            x = solution.variables[i]
            y = solution.variables[i + 1] 
             
            if i == 0:
                fitness1 += self.distance_to_warehouse[x]    
                fitness2 += self.cost_to_warehouse[x]                
                route_demand += self.demand_section[x-1]  
                
            route_demand += self.demand_section[y-1] 
            if y > (self.number_of_variables - self.num_vehicles):              
                route_demand = 0    
            if route_demand > self.vehicle_capacity:
                fitness1 += 99999999
                fitness2 += 99999999
            fitness1 += self.distance_matrix[x][y]       
            fitness2 += self.cost_matrix[x][y]
                     
        solution.objectives[0] = fitness1
        solution.objectives[1] = fitness2

        # ${f_{eval}(x) = f_{max} – f(x)}$, where ${f(x) = totaldistance(x) + \lambda overcapacity(x) + \mu overtime(x)}$      

        return solution
    
    
    def create_solution(self) -> PermutationSolution:
        new_solution = PermutationSolution(number_of_variables=self.number_of_variables,
                                           number_of_objectives=self.number_of_objectives)
        destination = [self.number_of_variables]
        random.seed(1)
        new_solution.variables = random.sample(range(1,self.number_of_variables), k=self.number_of_variables - 1) + destination
      
        return new_solution
    
    @property
    def number_of_cities(self):
        return self.number_of_variables

    def get_name(self):
        return 'CVRP'
