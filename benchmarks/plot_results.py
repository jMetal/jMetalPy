import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_CSV = os.path.join(os.path.dirname(__file__), 'results', 'nsgaii_distance_archive_results.csv')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

rows = []
with open(RESULTS_CSV) as f:
    r = csv.DictReader(f)
    for rec in r:
        rec['elapsed_seconds'] = float(rec['elapsed_seconds'])
        rec['front_size'] = int(rec['front_size'])
        rec['config_evaluations'] = int(rec['config_evaluations'])
        # optional fields
        if 'igd' in rec and rec['igd'] not in (None, '', 'None'):
            try:
                rec['igd'] = float(rec['igd'])
            except Exception:
                rec['igd'] = None
        else:
            rec['igd'] = None

        if 'hv' in rec and rec['hv'] not in (None, '', 'None'):
            try:
                rec['hv'] = float(rec['hv'])
            except Exception:
                rec['hv'] = None
        else:
            rec['hv'] = None

        rows.append(rec)

# Group by config
by = {}
for rec in rows:
    by.setdefault(rec['config_evaluations'], []).append(rec)

# Boxplot for elapsed_seconds
labels = []
data = []
for cfg in sorted(by):
    labels.append(str(cfg))
    data.append([r['elapsed_seconds'] for r in by[cfg]])

plt.figure(figsize=(6,4))
plt.boxplot(data, labels=labels, showmeans=True)
plt.ylabel('Elapsed seconds')
plt.xlabel('Max evaluations')
plt.title('NSGA-II elapsed time')
path1 = os.path.join(OUT_DIR, 'elapsed_seconds_boxplot.png')
plt.tight_layout()
plt.savefig(path1)
plt.close()

# Boxplot for front_size
labels = []
data = []
for cfg in sorted(by):
    labels.append(str(cfg))
    data.append([r['front_size'] for r in by[cfg]])

plt.figure(figsize=(6,4))
plt.boxplot(data, labels=labels, showmeans=True)
plt.ylabel('Front size')
plt.xlabel('Max evaluations')
plt.title('NSGA-II final front size')
path2 = os.path.join(OUT_DIR, 'front_size_boxplot.png')
plt.tight_layout()
plt.savefig(path2)
plt.close()

print('Plots saved:')
print(path1)
print(path2)

# If IGD/HV present, plot them
has_igd = any(r.get('igd') is not None for r in rows)
has_hv = any(r.get('hv') is not None for r in rows)

import statistics

if has_igd:
    labels = []
    data = []
    stats_txt = []
    for cfg in sorted(by):
        vals = [r['igd'] for r in by[cfg] if r.get('igd') is not None]
        labels.append(str(cfg))
        data.append(vals)
        if vals:
            stats_txt.append(f"{cfg}: mean={statistics.mean(vals):.6f}, std={statistics.stdev(vals) if len(vals)>1 else 0:.6f}")
        else:
            stats_txt.append(f"{cfg}: no data")

    plt.figure(figsize=(6,4))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel('IGD')
    plt.xlabel('Max evaluations')
    plt.title('NSGA-II IGD')
    path3 = os.path.join(OUT_DIR, 'igd_boxplot.png')
    plt.tight_layout()
    plt.savefig(path3)
    plt.close()
    print('Saved', path3)
    print('IGD stats:')
    for s in stats_txt:
        print(' ', s)

if has_hv:
    labels = []
    data = []
    stats_txt = []
    for cfg in sorted(by):
        vals = [r['hv'] for r in by[cfg] if r.get('hv') is not None]
        labels.append(str(cfg))
        data.append(vals)
        if vals:
            stats_txt.append(f"{cfg}: mean={statistics.mean(vals):.6f}, std={statistics.stdev(vals) if len(vals)>1 else 0:.6f}")
        else:
            stats_txt.append(f"{cfg}: no data")

    plt.figure(figsize=(6,4))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel('Hypervolume')
    plt.xlabel('Max evaluations')
    plt.title('NSGA-II Hypervolume')
    path4 = os.path.join(OUT_DIR, 'hv_boxplot.png')
    plt.tight_layout()
    plt.savefig(path4)
    plt.close()
    print('Saved', path4)
    print('HV stats:')
    for s in stats_txt:
        print(' ', s)
