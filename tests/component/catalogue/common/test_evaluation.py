"""Tests for the Evaluation component."""

from jmetal.component.catalogue.common.evaluation import (
    Evaluation,
    SequentialEvaluation,
    SequentialEvaluationWithArchive,
)
from jmetal.problem import ZDT1
from jmetal.problem.singleobjective.unconstrained import Sphere
from jmetal.util.archive import NonDominatedSolutionsArchive


class TestSequentialEvaluation:
    def test_evaluates_every_solution(self):
        problem = Sphere(number_of_variables=3)
        evaluation = SequentialEvaluation(problem)
        population = [problem.create_solution() for _ in range(5)]

        evaluated = evaluation.evaluate(population)

        assert all(solution.objectives[0] is not None for solution in evaluated)

    def test_reports_the_number_of_computed_evaluations(self):
        problem = Sphere()
        evaluation = SequentialEvaluation(problem)
        population = [problem.create_solution() for _ in range(7)]

        evaluation.evaluate(population)

        assert evaluation.computed_evaluations() == 7

    def test_exposes_the_wrapped_problem(self):
        problem = Sphere()
        evaluation = SequentialEvaluation(problem)

        assert evaluation.problem is problem

    def test_satisfies_the_evaluation_protocol(self):
        evaluation = SequentialEvaluation(Sphere())

        assert isinstance(evaluation, Evaluation)

    def test_accepts_an_injected_evaluator(self):
        class SpyEvaluator:
            def __init__(self):
                self.calls = 0

            def evaluate(self, solution_list, problem):
                self.calls += 1
                for solution in solution_list:
                    problem.evaluate(solution)
                return solution_list

        problem = Sphere(number_of_variables=2)
        spy = SpyEvaluator()
        evaluation = SequentialEvaluation(problem, evaluator=spy)
        population = [problem.create_solution() for _ in range(4)]

        evaluated = evaluation.evaluate(population)

        assert spy.calls == 1
        assert len(evaluated) == 4


class TestSequentialEvaluationWithArchive:
    def test_evaluates_every_solution(self):
        problem = ZDT1()
        evaluation = SequentialEvaluationWithArchive(problem, NonDominatedSolutionsArchive())
        population = [problem.create_solution() for _ in range(5)]

        evaluated = evaluation.evaluate(population)

        assert all(solution.objectives[0] is not None for solution in evaluated)

    def test_feeds_every_evaluated_solution_into_the_archive(self):
        problem = ZDT1()
        archive = NonDominatedSolutionsArchive()
        evaluation = SequentialEvaluationWithArchive(problem, archive)
        population = [problem.create_solution() for _ in range(20)]

        evaluation.evaluate(population)

        # Non-dominated archive may prune dominated duplicates, but with 20 random
        # ZDT1 solutions at least some should survive as non-dominated.
        assert archive.size() > 0
        assert archive.size() <= 20

    def test_archive_receives_copies_not_the_original_solutions(self):
        problem = ZDT1()
        archive = NonDominatedSolutionsArchive()
        evaluation = SequentialEvaluationWithArchive(problem, archive)
        population = [problem.create_solution() for _ in range(3)]

        evaluation.evaluate(population)

        for archived_solution in archive.solution_list:
            assert not any(archived_solution is p for p in population)

    def test_satisfies_the_evaluation_protocol(self):
        evaluation = SequentialEvaluationWithArchive(ZDT1(), NonDominatedSolutionsArchive())

        assert isinstance(evaluation, Evaluation)

    def test_still_reports_computed_evaluations(self):
        problem = ZDT1()
        evaluation = SequentialEvaluationWithArchive(problem, NonDominatedSolutionsArchive())
        population = [problem.create_solution() for _ in range(6)]

        evaluation.evaluate(population)

        assert evaluation.computed_evaluations() == 6
