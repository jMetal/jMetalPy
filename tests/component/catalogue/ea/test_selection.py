"""Tests for the Selection component."""

from jmetal.component.catalogue.ea.selection import Selection, TournamentSelection
from jmetal.operator.selection import TournamentSelection as TournamentSelectionOperator
from jmetal.problem.singleobjective.unconstrained import Sphere
from jmetal.util.comparator import ObjectiveComparator


class TestTournamentSelection:
    def _population(self, size: int = 10):
        problem = Sphere(number_of_variables=2)
        population = [problem.create_solution() for _ in range(size)]
        for i, solution in enumerate(population):
            solution.objectives[0] = float(i)
        return population

    def test_returns_a_mating_pool_of_the_requested_size(self):
        operator = TournamentSelectionOperator(tournament_size=2, comparator=ObjectiveComparator(0))
        selection = TournamentSelection(operator, mating_pool_size=6)

        mating_pool = selection.select(self._population())

        assert len(mating_pool) == 6

    def test_selected_solutions_come_from_the_population(self):
        operator = TournamentSelectionOperator(tournament_size=2, comparator=ObjectiveComparator(0))
        selection = TournamentSelection(operator, mating_pool_size=4)
        population = self._population()

        mating_pool = selection.select(population)

        assert all(solution in population for solution in mating_pool)

    def test_satisfies_the_selection_protocol(self):
        operator = TournamentSelectionOperator(tournament_size=2, comparator=ObjectiveComparator(0))
        selection = TournamentSelection(operator, mating_pool_size=4)

        assert isinstance(selection, Selection)
