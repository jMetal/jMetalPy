import os
import time
import csv
import numpy as np
from statistics import median

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.multiobjective.dtlz import DTLZ2
from jmetal.util.archive import DistanceBasedArchive
from jmetal.util.distance import DistanceMetric
from jmetal.util.evaluator import SequentialEvaluatorWithArchive
from jmetal.util.solution import read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.core.quality_indicator import HyperVolume

OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', '40k')
FRONTS_DIR = os.path.join(OUT_DIR, 'fronts')
PLOTS_DIR = os.path.join(OUT_DIR, 'plots')
os.makedirs(FRONTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, 'nsgaii_40k_hv_results.csv')

MAX_EVALS = 40000
SEEDS = list(range(10))

# prepare reference front (DTLZ2 3D if available)
ref_front = None
try:
    ref_front = read_solutions(filename='resources/reference_fronts/DTLZ2.3D.pf')
    ref_front = np.asarray([s.objectives for s in ref_front], dtype=float)
except Exception:
    ref_front = None

rows = []
# run experiments
for seed in SEEDS:
    print(f"Run seed={seed} (max_evals={MAX_EVALS})")
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
        termination_criterion=StoppingByEvaluations(max_evaluations=MAX_EVALS),
        population_evaluator=evaluator,
    )

    np.random.seed(seed)
    start = time.time()
    algorithm.run()
    elapsed = time.time() - start

    front = evaluator.get_archive().solution_list
    front_arr = np.asarray([s.objectives for s in front], dtype=float)

    front_path = os.path.join(FRONTS_DIR, f'front_{MAX_EVALS}_{seed}.npy')
    np.save(front_path, front_arr)

    hv = None
    try:
        # reference point slightly above 1.0 (DTLZ2 in [0,1])
        ref_pt = [1.1] * front_arr.shape[1]
        hv_calc = HyperVolume(reference_point=ref_pt)
        hv = float(hv_calc.compute(front_arr))
    except Exception as e:
        print('HV compute failed (moocore missing?):', e)
        hv = None

    row = {
        'seed': seed,
        'elapsed_seconds': elapsed,
        'evaluations': getattr(algorithm, 'evaluations', MAX_EVALS),
        'front_size': len(front),
        'front_path': front_path,
        'hv': hv,
    }
    rows.append(row)

# write CSV
fieldnames = ['seed','elapsed_seconds','evaluations','front_size','front_path','hv']
with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print('\nRuns complete. CSV:', CSV_PATH)

# Generate boxplot for HV and elapsed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

hvs = [r['hv'] for r in rows if r['hv'] is not None]
elapsed = [r['elapsed_seconds'] for r in rows]

plt.figure(figsize=(6,4))
plt.boxplot([hvs], labels=['HV'], showmeans=True)
plt.title('Hypervolume (10 runs, 40k evals)')
plt.ylabel('HV')
plt.tight_layout()
box_hv_path = os.path.join(PLOTS_DIR, 'hv_boxplot_40k.png')
plt.savefig(box_hv_path)
plt.close()
print('Saved HV boxplot:', box_hv_path)

plt.figure(figsize=(6,4))
plt.boxplot([elapsed], labels=['Elapsed s'], showmeans=True)
plt.title('Elapsed seconds (10 runs, 40k evals)')
plt.ylabel('Seconds')
plt.tight_layout()
box_time_path = os.path.join(PLOTS_DIR, 'elapsed_boxplot_40k.png')
plt.savefig(box_time_path)
plt.close()
print('Saved elapsed boxplot:', box_time_path)

# Plot the front corresponding to median HV (if HV computed)
if hvs:
    med_hv = median(hvs)
    # find run with hv closest to median
    idx = min(range(len(rows)), key=lambda i: abs((rows[i]['hv'] or 0) - med_hv))
    med_front = np.load(rows[idx]['front_path'])
    print('Median HV:', med_hv, 'selected seed:', rows[idx]['seed'], 'front_path:', rows[idx]['front_path'])

    # For 3-objective DTLZ2, plot 3D scatter
    if med_front.shape[1] == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(med_front[:,0], med_front[:,1], med_front[:,2], s=20, c='C0')
        ax.set_xlabel('f1')
        ax.set_ylabel('f2')
        ax.set_zlabel('f3')
        ax.set_title(f'Pareto front (median HV={med_hv:.6f})')
        front_plot_path = os.path.join(PLOTS_DIR, 'median_hv_front_40k.png')
        plt.tight_layout()
        plt.savefig(front_plot_path)
        plt.close()
        print('Saved median-front plot:', front_plot_path)
    else:
        # for 2-objective or other dims, do 2D scatter using first two objectives
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(med_front[:,0], med_front[:,1], s=20, c='C0')
        ax.set_xlabel('f1')
        ax.set_ylabel('f2')
        ax.set_title(f'Pareto front (median HV={med_hv:.6f})')
        front_plot_path = os.path.join(PLOTS_DIR, 'median_hv_front_40k_2d.png')
        plt.tight_layout()
        plt.savefig(front_plot_path)
        plt.close()
        print('Saved median-front plot:', front_plot_path)
else:
    print('No HV values computed; cannot select median front')

print('\nAll artifacts in:', OUT_DIR)
