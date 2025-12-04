"""The jmetal.operator package provides a collection of evolutionary operators for optimization algorithms.

This package contains implementations of various crossover, mutation, selection, and replacement operators
commonly used in evolutionary computation. These operators are designed to work with the solution
representations defined in jmetal.core.solution.

The main categories of operators are:

1. Crossover (jmetal.operator.crossover):
   - SBXCrossover: Simulated Binary Crossover for real-valued solutions
   - BLXAlphaCrossover: Blend crossover with alpha parameter
   - PMXCrossover: Partially Mapped Crossover for permutation problems
   - SPXCrossover: Single-point crossover for binary solutions
   - And more specialized crossover operators

2. Mutation (jmetal.operator.mutation):
   - PolynomialMutation: Polynomial mutation for real-valued solutions
   - BitFlipMutation: Simple bit-flip mutation for binary solutions
   - PermutationSwapMutation: Swap mutation for permutation problems
   - And other mutation operators with different characteristics

3. Selection (jmetal.operator.selection):
   - BinaryTournamentSelection: Tournament selection with size 2
   - TournamentSelection: K-ary tournament selection with configurable size
   - NaryRandomSolutionSelection: Random selection of N solutions
   - BestSolutionSelection: Selects the best solutions from a list
   - And other selection strategies

4. Replacement (jmetal.operator.replacement):
   - RankingAndCrowdingDistanceReplacement: Used in NSGA-II
   - And other replacement strategies

Example usage:
    >>> from jmetal.algorithm.multiobjective import NSGAII
    >>> from jmetal.operator import SBXCrossover, PolynomialMutation
    >>> from jmetal.problem import ZDT1
    >>>
    >>> problem = ZDT1()
    >>> algorithm = NSGAII(
    ...     problem=problem,
    ...     population_size=100,
    ...     offspring_population_size=100,
    ...     mutation=PolynomialMutation(probability=1.0/problem.number_of_variables, distribution_index=20),
    ...     crossover=SBXCrossover(probability=0.9, distribution_index=20)
    ... )
"""

from .crossover import (
    BLXAlphaCrossover,
    BLXAlphaBetaCrossover,
    CXCrossover,
    DifferentialEvolutionCrossover,
    IntegerSBXCrossover,
    NullCrossover,
    PMXCrossover,
    SBXCrossover,
    SPXCrossover,
)
from .mutation import (
    BitFlipMutation,
    IntegerPolynomialMutation,
    NullMutation,
    PermutationSwapMutation,
    PolynomialMutation,
    ScrambleMutation,
    SimpleRandomMutation,
    UniformMutation,
)
from .replacement import (
    RankingAndCrowdingDistanceReplacement,
)
from .selection import (
    BestSolutionSelection,
    BinaryTournament2Selection,
    BinaryTournamentSelection,
    NaryRandomSolutionSelection,
    RandomSelection,
    RankingAndCrowdingDistanceSelection,
    RouletteWheelSelection,
    TournamentSelection,
)

__all__ = [
    "BLXAlphaCrossover",
    "BLXAlphaBetaCrossover",
    "CXCrossover",
    "DifferentialEvolutionCrossover",
    "IntegerSBXCrossover",
    "NullCrossover",
    "PMXCrossover",
    "SBXCrossover",
    "SPXCrossover",
    "BitFlipMutation",
    "IntegerPolynomialMutation",
    "NullMutation",
    "PermutationSwapMutation",
    "PolynomialMutation",
    "ScrambleMutation",
    "SimpleRandomMutation",
    "UniformMutation",
    "BestSolutionSelection",
    "BinaryTournament2Selection",
    "BinaryTournamentSelection",
    "NaryRandomSolutionSelection",
    "RandomSelection",
    "RankingAndCrowdingDistanceSelection",
    "RouletteWheelSelection",
    "TournamentSelection",
    "RankingAndCrowdingDistanceReplacement",
]
