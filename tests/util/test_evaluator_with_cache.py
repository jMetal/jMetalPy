import copy

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.evaluator import Evaluator, EvaluatorWithCache


class CountingProblem(FloatProblem):
    def __init__(self) -> None:
        super().__init__()
        self.lower_bound = [0.0, 0.0]
        self.upper_bound = [10.0, 10.0]

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 1

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        solution.objectives[0] = sum(solution.variables)
        solution.constraints[0] = solution.variables[0] - solution.variables[1]
        return solution

    def name(self) -> str:
        return "Counting problem"


class CountingEvaluator(Evaluator[FloatSolution]):
    def __init__(self, return_copies: bool = False) -> None:
        self.evaluated_variables: list[list[list[float]]] = []
        self.return_copies = return_copies

    def evaluate(
        self, solution_list: list[FloatSolution], problem: FloatProblem
    ) -> list[FloatSolution]:
        self.evaluated_variables.append([solution.variables for solution in solution_list])
        evaluated_solutions = []
        for solution in solution_list:
            evaluated_solution = copy.copy(solution) if self.return_copies else solution
            evaluated_solutions.append(problem.evaluate(evaluated_solution))

        return evaluated_solutions


def solution(problem: CountingProblem, variables: list[float]) -> FloatSolution:
    result = problem.create_solution()
    result.variables = variables
    return result


class TestEvaluatorWithCache:
    def test_should_evaluate_each_distinct_solution_once_and_preserve_order(self) -> None:
        problem = CountingProblem()
        base_evaluator = CountingEvaluator()
        evaluator = EvaluatorWithCache(base_evaluator)
        solutions = [
            solution(problem, [1.0, 2.0]),
            solution(problem, [3.0, 1.0]),
            solution(problem, [1.0, 2.0]),
        ]

        result = evaluator.evaluate(solutions, problem)

        assert result is solutions
        assert base_evaluator.evaluated_variables == [[[1.0, 2.0], [3.0, 1.0]]]
        assert [item.variables for item in result] == [
            [1.0, 2.0],
            [3.0, 1.0],
            [1.0, 2.0],
        ]
        assert [item.objectives for item in result] == [[3.0], [4.0], [3.0]]
        assert [item.constraints for item in result] == [[-1.0], [2.0], [-1.0]]

    def test_should_reuse_objectives_and_constraints_across_calls(self) -> None:
        problem = CountingProblem()
        base_evaluator = CountingEvaluator()
        evaluator = EvaluatorWithCache(base_evaluator)
        first = solution(problem, [2.0, 5.0])
        evaluator.evaluate([first], problem)
        first.objectives[0] = 99.0
        first.constraints[0] = 99.0
        cached = solution(problem, [2.0, 5.0])

        evaluator.evaluate([cached], problem)

        assert len(base_evaluator.evaluated_variables) == 1
        assert cached.objectives == [7.0]
        assert cached.constraints == [-3.0]

    def test_should_copy_results_returned_by_parallel_style_evaluator(self) -> None:
        problem = CountingProblem()
        base_evaluator = CountingEvaluator(return_copies=True)
        evaluator = EvaluatorWithCache(base_evaluator)
        unevaluated = solution(problem, [4.0, 1.0])

        result = evaluator.evaluate([unevaluated], problem)

        assert result == [unevaluated]
        assert result[0] is unevaluated
        assert unevaluated.objectives == [5.0]
        assert unevaluated.constraints == [3.0]

    def test_should_clear_cache_when_problem_instance_changes(self) -> None:
        first_problem = CountingProblem()
        second_problem = CountingProblem()
        base_evaluator = CountingEvaluator()
        evaluator = EvaluatorWithCache(base_evaluator)

        evaluator.evaluate([solution(first_problem, [1.0, 1.0])], first_problem)
        evaluator.evaluate([solution(second_problem, [1.0, 1.0])], second_problem)

        assert base_evaluator.evaluated_variables == [
            [[1.0, 1.0]],
            [[1.0, 1.0]],
        ]

    def test_should_return_empty_list_without_calling_base_evaluator(self) -> None:
        problem = CountingProblem()
        base_evaluator = CountingEvaluator()
        evaluator = EvaluatorWithCache(base_evaluator)

        result = evaluator.evaluate([], problem)

        assert result == []
        assert base_evaluator.evaluated_variables == []
