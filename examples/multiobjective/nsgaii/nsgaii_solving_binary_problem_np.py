from __future__ import annotations

import time
import numpy as np
import os
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.solution import BinarySolutionNP
from jmetal.operator.crossover import SPXNPCrossover
from jmetal.operator.mutation import BitFlipNPMutation
from jmetal.core.problem import BinaryProblem
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.observer import ProgressBarObserver

class OneZeroMaxNP(BinaryProblem):
    """
    Optimized version of OneZeroMax problem using NumPy for better performance.
    """
    def __init__(self, number_of_bits: int = 10000):
        super(OneZeroMaxNP, self).__init__()
        self._number_of_variables = number_of_bits
        self._number_of_objectives = 2
        self._number_of_constraints = 0
        
    @property
    def number_of_variables(self) -> int:
        return self._number_of_variables
        
    @property
    def number_of_objectives(self) -> int:
        return self._number_of_objectives
        
    @property
    def number_of_constraints(self) -> int:
        return self._number_of_constraints
        
    @property
    def name(self) -> str:
        return 'OneZeroMaxNP'

    def evaluate(self, solution):
        # Count ones (first objective is to maximize the number of ones)
        ones = np.sum(solution.bits)
        
        # Count zeros (second objective is to maximize the number of zeros)
        zeros = solution.number_of_variables - ones
        
        # Set the objectives (negative because we want to maximize)
        solution.objectives[0] = -ones
        solution.objectives[1] = -zeros
        
        return solution

    def create_solution(self):
        # Create a new solution with the specified number of variables
        solution = BinarySolutionNP(
            number_of_variables=self.number_of_variables,
            number_of_objectives=self.number_of_objectives,
            number_of_constraints=self.number_of_constraints
        )
        
        # Initialize with random bits
        solution.bits[:] = np.random.random(self.number_of_variables) < 0.5
        
        return solution

    def get_name(self) -> str:
        return 'OneZeroMaxNP'

def run_optimization():
    # Configuration
    number_of_bits = 10000  # 10,000 bits for a more challenging problem
    max_evaluations = 30000
    population_size = 100
    
    # Create the problem
    problem = OneZeroMaxNP(number_of_bits)
    
    # Create the operators
    crossover_probability = 1.0
    crossover = SPXNPCrossover(probability=crossover_probability)
    
    mutation_probability = 1.0 / number_of_bits
    mutation = BitFlipNPMutation(probability=mutation_probability)
    
    # Create the algorithm
    algorithm = NSGAII(
        problem=problem,
        population_size=population_size,
        offspring_population_size=population_size,
        crossover=crossover,
        mutation=mutation,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )
    
    # Add progress bar observer
    progress_bar = ProgressBarObserver(max=max_evaluations)
    algorithm.observable.register(progress_bar)
    
    # Run the algorithm and measure time
    print(f"Running NSGA-II with BinarySolutionNP on {number_of_bits} bits...")
    start_time = time.time()
    algorithm.run()
    total_time = time.time() - start_time
    
    # Get the results
    front = algorithm.result()
    
    # Create output directory if it doesn't exist
    output_dir = "resources/NSGAII_NP"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the results
    print(f"\nSaving results to {output_dir} directory...")
    print_function_values_to_file(front, f"{output_dir}/FUN.NSGAII.OneZeroMaxNP")
    print_variables_to_file(front, f"{output_dir}/VAR.NSGAII.OneZeroMaxNP")
    
    # Print summary
    print("\nAlgorithm execution finished!")
    print(f"Algorithm: {algorithm.get_name()} (NumPy-optimized)")
    print(f"Problem: {problem.get_name()}")
    print(f"Population size: {population_size}")
    print(f"Number of evaluations: {algorithm.evaluations}")
    print(f"Computing time: {total_time:.3f} seconds")
    
    # Print some statistics
    if front:
        print(f"\nNumber of non-dominated solutions: {len(front)}")
        print("First solution objectives:", front[0].objectives)
        print("Number of 1s:", front[0].cardinality())
        print("Number of 0s:", number_of_bits - front[0].cardinality())
    
    return total_time

if __name__ == "__main__":
    # Run the optimization and measure time
    np_time = run_optimization()
    
    # Print comparison with original version (from previous run)
    print("\n--- Performance Comparison ---")
    print(f"Original version: 26.13 seconds (from previous run)")
    print(f"NumPy-optimized:  {np_time:.2f} seconds")
    print(f"Speedup:          {26.13/np_time:.2f}x")
