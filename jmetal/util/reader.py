# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:15:45 2021

@author: María Fdez Hijano
"""


import math
import re


class Reader():
     
    def __init__(self, filename, num_vehicles):
        self.filename = filename
        self.num_vehicles = num_vehicles
        
    def read_from_file(self):
        """
        This function reads a CVRP Problem instance from a file.
    
        :param filename: File which describes the instance.
        :type filename: str.
        """
        
        
        if self.filename is None:
            raise FileNotFoundError('Filename can not be None')
    
        with open(self.filename) as file:    
            lines = file.readlines()
            data = [line.lstrip() for line in lines if line != ""]  
            
            dimension = re.compile(r'[^\d]+')
            dummy_nodes = self.num_vehicles
                        
            for item in data:      
                if item.startswith('DIMENSION'):
                        dimension = int(dimension.sub('', item)) + dummy_nodes                                
                if item.startswith("NODE_COORD_SECTION"):                   
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
                                    
            return distance_matrix, distance_to_warehouse
           
            
    def read_demand_from_file(self):
        """
        This function reads demand from a file.
    
        :param filename: File which describes the instance.
        :type filename: str.
        """       
        if self.filename is None:
            raise FileNotFoundError('Filename can not be None')
    
        with open(self.filename) as file:    
            lines = file.readlines()
            data = [line.lstrip() for line in lines if line != ""]           
    
            dimension = re.compile(r'[^\d]+')        
            dummy_nodes = self.num_vehicles
                     
            for item in data: 
                if item.startswith('DIMENSION'):
                    dimension = int(dimension.sub('', item)) + dummy_nodes
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
                        
        return demand_section
    
    def read_dimension_and_capacity_from_file(self):
        """
        This function reads dimension and capacity from a file.
    
        :param filename: File which describes the instance.
        :type filename: str.
        """
        if self.filename is None:
            raise FileNotFoundError('Filename can not be None')       
            
        with open(self.filename) as file:    
            lines = file.readlines()
            data = [line.lstrip() for line in lines if line != ""]  
            dimension = re.compile(r'[^\d]+') 
            dummy_nodes = self.num_vehicles
            vehicle_capacity = re.compile(r'[^\d]+')      
    
        for item in data:
            if item.startswith('DIMENSION'):
                dimension = int(dimension.sub('', item)) + dummy_nodes
            elif item.startswith('CAPACITY'):
                vehicle_capacity = int(vehicle_capacity.sub('', item))                             
          
        return dimension - 1, vehicle_capacity