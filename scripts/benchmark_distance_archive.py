import time
import numpy as np
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.multiobjective.dtlz import DTLZ2
from jmetal.util.archive import DistanceBasedArchive
from jmetal.util.distance import DistanceMetric
from jmetal.util.evaluator import SequentialEvaluatorWithArchive
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file, read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations


def run_experiment(use_vectorized: bool, max_evaluations: int, seed: int = 1):
    problem = DTLZ2()
    problem.reference_front = read_solutions(filename="resources/reference_fronts/DTLZ2.3D.pf")

    archive = DistanceBasedArchive(maximum_size=100, metric=DistanceMetric.L2_SQUARED, use_vectorized=use_vectorized)
    evaluator = SequentialEvaluatorWithArchive(archive)

    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        population_evaluator=evaluator,
        # seed handling may be internal; ensure reproducibility by setting numpy seed
    )

    np.random.seed(seed)
    start = time.time()
    algorithm.run()
    elapsed = time.time() - start

    front = evaluator.get_archive().solution_list
    # Write front to file for later analysis
    label = f"vectorized_{use_vectorized}_evals_{max_evaluations}_seed_{seed}"
    fun_file = "FUN." + label
    var_file = "VAR." + label
    print_function_values_to_file(front, fun_file)
    print_variables_to_file(front, var_file)

    return elapsed, front, fun_file


def load_front(path: str):
    sols = read_solutions(path)
    # convert to numpy array of objectives
    arr = np.array([s.objectives for s in sols])
    return arr


def average_distance_to_reference(front_arr: np.ndarray, ref_arr: np.ndarray):
    # For each point in front_arr, compute min distance to ref_arr, then average
    if front_arr.size == 0 or ref_arr.size == 0:
        return float('inf')
    dists = []
    for p in front_arr:
        d = np.sqrt(((ref_arr - p) ** 2).sum(axis=1))
        dists.append(d.min())
    return float(np.mean(dists))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-evals', type=int, default=2000, help='max evaluations for quick benchmark')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    ref = read_solutions(filename="resources/reference_fronts/DTLZ2.3D.pf")
    ref_arr = np.array([s.objectives for s in ref])

    results = {}
    for use_vec in (True, False):
        print(f"Running experiment use_vectorized={use_vec} max_evals={args.max_evals}")
        elapsed, front, fun_file = run_experiment(use_vec, args.max_evals, seed=args.seed)
        front_arr = np.array([s.objectives for s in front])
        avg_dist = average_distance_to_reference(front_arr, ref_arr)
        print(f"use_vectorized={use_vec} time={elapsed:.2f}s avg_dist_to_ref={avg_dist:.6f} saved_front={fun_file}")
        results[use_vec] = dict(time=elapsed, avg_dist=avg_dist, fun_file=fun_file)

    print("Summary:")
    for use_vec, r in results.items():
        print(f"use_vectorized={use_vec}: time={r['time']:.2f}s avg_dist={r['avg_dist']:.6f} file={r['fun_file']}")
