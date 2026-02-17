import time
import csv
import os
import math
import numpy as np
from statistics import mean, stdev

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.multiobjective.dtlz import DTLZ2
from jmetal.util.archive import DistanceBasedArchive
from jmetal.util.distance import DistanceMetric
from jmetal.util.evaluator import SequentialEvaluatorWithArchive
from jmetal.util.solution import read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

# Optional metrics helpers (IGD/HV) - best-effort if available
try:
    from jmetal.util.quality_indicator import InvertedGenerationalDistance, Hypervolume
    HAVE_QI = True
except Exception:
    HAVE_QI = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "nsgaii_distance_archive_results.csv")

PROBLEM = DTLZ2()
REF_FRONT = PROBLEM.reference_front if hasattr(PROBLEM, 'reference_front') and PROBLEM.reference_front else None
if not REF_FRONT:
    try:
        PROBLEM.reference_front = read_solutions("resources/reference_fronts/DTLZ2.3D.pf")
        REF_FRONT = PROBLEM.reference_front
    except Exception:
        REF_FRONT = None

CONFIGS = [2000, 20000]
N_REPEATS = 10

fieldnames = [
    "config_evaluations",
    "seed",
    "elapsed_seconds",
    "evaluations",
    "front_size",
    "archive_utilization",
]
if HAVE_QI:
    fieldnames += ["igd", "hv"]

with open(CSV_PATH, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for max_evals in CONFIGS:
        for seed in range(N_REPEATS):
            print(f"Running config={max_evals}, seed={seed}")
            # Build algorithm
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
            # set seed where applicable
            np.random.seed(seed)

            start = time.time()
            algorithm.run()
            elapsed = time.time() - start

            front = evaluator.get_archive().solution_list
            front_size = len(front)
            archive_util = f"{front_size}/{archive.maximum_size} ({front_size/archive.maximum_size*100:.1f}%)"

            row = {
                "config_evaluations": max_evals,
                "seed": seed,
                "elapsed_seconds": elapsed,
                "evaluations": getattr(algorithm, 'evaluations', max_evals),
                "front_size": front_size,
                "archive_utilization": archive_util,
            }

            if HAVE_QI and REF_FRONT:
                try:
                    igd = InvertedGenerationalDistance(REF_FRONT).compute(front)
                except Exception:
                    igd = None
                try:
                    hv = Hypervolume(reference_point=[1.1] * len(front[0].objectives)).compute(front)
                except Exception:
                    hv = None
                row.update({"igd": igd, "hv": hv})

            writer.writerow(row)
            csvfile.flush()

print(f"Benchmark finished. Results: {CSV_PATH}")
print("Now you can inspect CSV or ask me to generate boxplots from it.")
