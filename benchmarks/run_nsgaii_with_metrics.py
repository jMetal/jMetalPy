import os
import time
import csv
import numpy as np
from types import SimpleNamespace
from statistics import mean, stdev

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.multiobjective.dtlz import DTLZ2
from jmetal.util.archive import DistanceBasedArchive
from jmetal.util.distance import DistanceMetric
from jmetal.util.evaluator import SequentialEvaluatorWithArchive
from jmetal.util.solution import read_solutions
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.operator.selection import RankingAndFitnessSelection

from jmetal.util.termination_criterion import StoppingByEvaluations

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
FRONTS_DIR = os.path.join(OUT_DIR, "fronts")
os.makedirs(FRONTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "nsgaii_distance_archive_results_with_metrics.csv")

CONFIGS = [2000, 20000]
N_REPEATS = 10

# IGD implementation: average distance from reference front points to nearest point in front

def compute_igd(reference_front, front):
    # reference_front: array (m x d)
    # front: array (n x d)
    if front.size == 0:
        return float('inf')
    dists = []
    for r in reference_front:
        # Euclidean distance to nearest point
        diff = front - r
        dist2 = np.sum(diff * diff, axis=1)
        d = np.sqrt(np.min(dist2))
        dists.append(d)
    return float(np.mean(dists))

# Hypervolume: use RankingAndFitnessSelection.hypesub via compute_hypervol_fitness_values

def compute_hypervolume(front, ref_point):
    # front: numpy array (n x d)
    # ref_point: list
    if front.size == 0:
        return 0.0
    # Convert to the Solution-like objects required by compute_hypervol_fitness_values
    class FakeSol:
        def __init__(self, objectives):
            self.objectives = objectives
            self.attributes = {}
    pop = [FakeSol(list(p)) for p in front]
    ref = SimpleNamespace(objectives=list(ref_point))
    selector = RankingAndFitnessSelection(max_population_size=len(pop), reference_point=ref)
    # compute_hypervol_fitness_values stores contributions in attributes['fitness'] and returns population
    selector.compute_hypervol_fitness_values(pop, ref, k=-1)
    contributions = [s.attributes.get('fitness', 0.0) for s in pop]
    return float(sum(contributions))

# Load reference front
ref_front = None
try:
    rf = read_solutions("resources/reference_fronts/DTLZ2.3D.pf")
    ref_front = np.asarray([s.objectives for s in rf], dtype=float)
except Exception:
    ref_front = None

with open(CSV_PATH, "w", newline="") as csvfile:
    fieldnames = ["config_evaluations", "seed", "elapsed_seconds", "evaluations", "front_size", "igd", "hv"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for max_evals in CONFIGS:
        for seed in range(N_REPEATS):
            print(f"Running config={max_evals}, seed={seed}")
            problem = DTLZ2()
            try:
                problem.reference_front = read_solutions(filename="resources/reference_fronts/DTLZ2.3D.pf")
            except Exception:
                pass

            archive = DistanceBasedArchive(maximum_size=100, metric=DistanceMetric.L2_SQUARED, use_vectorized=True)
            evaluator = SequentialEvaluatorWithArchive(archive)

            algorithm = NSGAII(
                problem=problem,
                population_size=100,
                offspring_population_size=100,
                mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evals),
                population_evaluator=evaluator,
            )

            np.random.seed(seed)

            start = time.time()
            algorithm.run()
            elapsed = time.time() - start

            front = evaluator.get_archive().solution_list
            front_array = np.asarray([s.objectives for s in front], dtype=float)

            # Save front
            front_path = os.path.join(FRONTS_DIR, f"front_{max_evals}_{seed}.npy")
            np.save(front_path, front_array)

            igd = None
            hv = None
            if ref_front is not None and ref_front.size > 0 and front_array.size > 0:
                try:
                    igd = compute_igd(ref_front, front_array)
                except Exception:
                    igd = None
                try:
                    # choose reference point slightly worse than 1.0 range
                    hv = compute_hypervolume(front_array, [1.1] * front_array.shape[1])
                except Exception:
                    hv = None

            row = {
                "config_evaluations": max_evals,
                "seed": seed,
                "elapsed_seconds": elapsed,
                "evaluations": getattr(algorithm, 'evaluations', max_evals),
                "front_size": len(front),
                "igd": igd,
                "hv": hv,
            }
            writer.writerow(row)
            csvfile.flush()

print(f"Benchmark with metrics finished. Results: {CSV_PATH}")
print(f"Front files saved in: {FRONTS_DIR}")
