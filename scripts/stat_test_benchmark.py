#!/usr/bin/env python3
import csv
import math
import statistics

CSV_FILE = 'benchmark_20000_10seeds_metrics.csv'
OUT_FILE = 'benchmark_20000_stats.txt'

igd_true = []
igd_false = []

with open(CSV_FILE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        mode = row['mode']
        igd = None
        try:
            igd = float(row['igd']) if row['igd'] and row['igd'].lower()!='none' else None
        except Exception:
            igd = None
        if igd is None:
            continue
        if mode == 'vectorized_True':
            igd_true.append(igd)
        elif mode == 'vectorized_False':
            igd_false.append(igd)

# Expect paired samples of equal length
if len(igd_true) != len(igd_false):
    print('Warning: unequal sample sizes:', len(igd_true), len(igd_false))

n = min(len(igd_true), len(igd_false))
igd_true = igd_true[:n]
igd_false = igd_false[:n]

diffs = [a - b for a, b in zip(igd_true, igd_false)]
mean_diff = statistics.mean(diffs) if diffs else float('nan')
std_diff = statistics.stdev(diffs) if len(diffs) > 1 else float('nan')

results = {}

# Try scipy tests; if not available, skip
try:
    from scipy import stats
    t_stat, t_p = stats.ttest_rel(igd_true, igd_false)
    try:
        w_stat, w_p = stats.wilcoxon(igd_true, igd_false)
    except Exception as e:
        w_stat, w_p = (None, None)
    results['t_stat'] = t_stat
    results['t_p'] = t_p
    results['w_stat'] = w_stat
    results['w_p'] = w_p
    results['scipy'] = True
except Exception as e:
    # Fallback: compute paired t manually
    import math
    mean_t = statistics.mean(igd_true) - statistics.mean(igd_false)
    sd1 = statistics.stdev(igd_true) if len(igd_true) > 1 else 0.0
    sd2 = statistics.stdev(igd_false) if len(igd_false) > 1 else 0.0
    # paired t requires std of differences
    sd_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    se = sd_diff / math.sqrt(n) if n > 0 else float('nan')
    t_stat = mean_diff / se if se and not math.isnan(se) else float('nan')
    t_p = None
    results['t_stat'] = t_stat
    results['t_p'] = t_p
    results['w_stat'] = None
    results['w_p'] = None
    results['scipy'] = False

# Cohen's d for paired samples
try:
    mean_a = statistics.mean(igd_true)
    mean_b = statistics.mean(igd_false)
    sd_a = statistics.stdev(igd_true)
    sd_b = statistics.stdev(igd_false)
    # use sd of differences for paired Cohen's d
    sd_diff = statistics.stdev(diffs) if len(diffs) > 1 else float('nan')
    cohens_d = mean_diff / sd_diff if sd_diff and not math.isnan(sd_diff) else float('nan')
except Exception:
    cohens_d = float('nan')

# Write report
with open(OUT_FILE, 'w') as f:
    f.write('Statistical comparison of IGD (vectorized_True vs vectorized_False)\n')
    f.write(f'n = {n}\n')
    f.write(f'mean(vectorized_True) = {statistics.mean(igd_true) if igd_true else float('nan')}\n')
    f.write(f'mean(vectorized_False) = {statistics.mean(igd_false) if igd_false else float('nan')}\n')
    f.write(f'mean(difference True-False) = {mean_diff}\n')
    f.write(f'std(difference) = {std_diff}\n')
    f.write('\n')
    if results.get('scipy'):
        f.write(f'Paired t-test: t = {results["t_stat"]}, p = {results["t_p"]}\n')
        if results.get('w_stat') is not None:
            f.write(f'Wilcoxon signed-rank: stat = {results["w_stat"]}, p = {results["w_p"]}\n')
        else:
            f.write('Wilcoxon: not available (error computing)\n')
    else:
        f.write(f'Paired t-test (approx): t = {results["t_stat"]} (scipy missing, p not computed)\n')
        f.write('Wilcoxon: skipped (scipy missing)\n')
    f.write(f"Cohen's d (paired) = {cohens_d}\n")

# Also print to stdout
with open(OUT_FILE) as f:
    print(f.read())
