#!/usr/bin/env python3
import csv
import math
import matplotlib.pyplot as plt

CSV_FILE = 'benchmark_20000_10seeds_metrics.csv'
OUT_PNG = 'benchmark_20000_boxplot.png'

modes = []
igd_values = {'vectorized_True': [], 'vectorized_False': []}
hv_values = {'vectorized_True': [], 'vectorized_False': []}

with open(CSV_FILE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        mode = row['mode']
        try:
            igd = float(row['igd']) if row['igd'] and row['igd'].lower()!='none' else None
        except Exception:
            igd = None
        try:
            hv = float(row['hv']) if row['hv'] and row['hv'].lower()!='none' else None
        except Exception:
            hv = None
        if igd is not None:
            igd_values[mode].append(igd)
        if hv is not None:
            hv_values[mode].append(hv)

# Prepare boxplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# IGD boxplot
data_igd = [igd_values['vectorized_True'], igd_values['vectorized_False']]
axes[0].boxplot(data_igd, labels=['vectorized_True', 'vectorized_False'])
axes[0].set_title('IGD (lower is better)')
axes[0].set_ylabel('IGD')

# HV boxplot
data_hv = [hv_values['vectorized_True'], hv_values['vectorized_False']]
axes[1].boxplot(data_hv, labels=['vectorized_True', 'vectorized_False'])
axes[1].set_title('Hypervolume (higher is better)')
axes[1].set_ylabel('HV')

plt.tight_layout()
plt.savefig(OUT_PNG)
print('Saved boxplot to', OUT_PNG)
