import os
import time
import csv
import numpy as np
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.multiobjective.dtlz import DTLZ2
from jmetal.util.archive import DistanceBasedArchive
from jmetal.util.distance import DistanceMetric
from jmetal.util.evaluator import SequentialEvaluatorWithArchive
from jmetal.util.solution import read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

# quality indicators
from jmetal.core.quality_indicator import InvertedGenerationalDistance, HyperVolume

OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
FRONTS_DIR = os.path.join(OUT_DIR, 'fronts')
os.makedirs(FRONTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, 'nsgaii_distance_archive_results.csv')

CONFIGS = [2000, 20000]
SEEDS = [0, 1]

# load reference front
ref_front = None
try:
    ref_front = read_solutions(filename='resources/reference_fronts/DTLZ2.3D.pf')
    ref_front = np.asarray([s.objectives for s in ref_front], dtype=float)
except Exception:
    ref_front = None

rows = []
# We'll append rows to CSV; read existing to keep ordering
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

new_rows = []
for max_evals in CONFIGS:
    for seed in SEEDS:
        print(f"Running sample config={max_evals}, seed={seed}")
        problem = DTLZ2()
        try:
            problem.reference_front = read_solutions(filename='resources/reference_fronts/DTLZ2.3D.pf')
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
        front_arr = np.asarray([s.objectives for s in front], dtype=float)

        # save front
        front_path = os.path.join(FRONTS_DIR, f'front_{max_evals}_{seed}.npy')
        np.save(front_path, front_arr)

        igd = None
        hv = None
        if ref_front is not None and front_arr.size > 0:
            try:
                igd_calc = InvertedGenerationalDistance(ref_front)
                igd = igd_calc.compute(front_arr)
            except Exception as e:
                print('IGD compute failed:', e)
                igd = None
            try:
                # HyperVolume expects a reference point; use 1.1 per objective (DTLZ2 has range [0,1])
                ref_point = [1.1] * front_arr.shape[1]
                hv_calc = HyperVolume(reference_point=ref_point)
                hv = hv_calc.compute(front_arr)
            except Exception as e:
                print('HV compute failed (moocore may be missing):', e)
                hv = None

        row = {
            'config_evaluations': max_evals,
            'seed': seed,
            'elapsed_seconds': elapsed,
            'evaluations': getattr(algorithm, 'evaluations', max_evals),
            'front_size': len(front),
            'archive_utilization': f"{len(front)}/{archive.maximum_size} ({len(front)/archive.maximum_size*100:.1f}%)",
            'igd': igd,
            'hv': hv,
            'front_path': front_path,
        }
        new_rows.append(row)

# Append to CSV file (add IGD/HV columns if needed)
fieldnames = ['config_evaluations','seed','elapsed_seconds','evaluations','front_size','archive_utilization','igd','hv','front_path']
write_header = not os.path.exists(CSV_PATH)
with open(CSV_PATH, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    for r in new_rows:
        writer.writerow(r)

print('Sample runs finished. Front files:')
for r in new_rows:
    print(r['front_path'], 'IGD=', r['igd'], 'HV=', r['hv'])
