"""Tests for the Evaluation component."""

from jmetal.component.catalogue.common.evaluation import Evaluation, SequentialEvaluation
from jmetal.problem.singleobjective.unconstrained import Sphere


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
