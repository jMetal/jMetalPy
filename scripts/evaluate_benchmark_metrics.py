#!/usr/bin/env python3
import os
import sys
import numpy as np
from math import inf

SEEDS = range(1, 11)
MODES = ["vectorized_True", "vectorized_False"]
DEFAULT_MAX_EVALS = 2000
FUN_TEMPLATE = "FUN.{mode}_evals_{max_evals}_seed_{seed}"
REF_PATHS = [
    "resources/reference_fronts/DTLZ2.3D.pf",
    "resources/reference_fronts/DTLZ2.3D.pf.txt",
]


def load_reference():
    for p in REF_PATHS:
        if os.path.exists(p):
            return np.loadtxt(p)
    raise FileNotFoundError("Reference front not found in expected paths: %s" % REF_PATHS)


def load_front(path):
    if not os.path.exists(path):
        return None
    try:
        return np.loadtxt(path)
    except Exception:
        # try to read as whitespace separated with possible header
        data = []
        with open(path) as f:
            for line in f:
                line=line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                try:
                    nums = [float(x) for x in parts]
                    data.append(nums)
                except Exception:
                    continue
        if not data:
            return None
        return np.array(data)


def igd(reference, front):
    # reference: (m, d), front: (n, d)
    if front is None or len(front)==0:
        return float('inf')
    ref = np.atleast_2d(reference)
    fr = np.atleast_2d(front)
    dists = []
    for r in ref:
        d = np.linalg.norm(fr - r, axis=1)
        dists.append(d.min())
    return float(np.mean(dists))


def try_hypervolume(front):
    # Prefer pygmo if available for hypervolume; otherwise fall back to internal jMetalPy HyperVolume
    if front is None or len(front) == 0:
        return 0.0, "ok"
    try:
        import pygmo as pg  # type: ignore
    except Exception:
        # fallback to jMetalPy HyperVolume
        try:
            from jmetal.core.quality_indicator import HyperVolume

            ref_point = np.max(front, axis=0) * 1.1
            hv = HyperVolume(ref_point.tolist())
            val = hv.compute(np.atleast_2d(front))
            return float(val), "jmetal_hv"
        except Exception as e:
            return None, f"error:{e}"
    else:
        try:
            ref_point = np.max(front, axis=0) * 1.1
            hv = pg.hypervolume(front.tolist())
            val = hv.compute(ref_point.tolist())
            return float(val), "pygmo"
        except Exception as e:
            return None, f"error:{e}"


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate benchmark metrics (IGD, HV)')
    parser.add_argument('--max-evals', type=int, default=DEFAULT_MAX_EVALS, help='max evaluations used in filenames')
    args = parser.parse_args()

    max_evals = args.max_evals
    reference = load_reference()
    results = []
    missing = []

    for seed in SEEDS:
        for mode in MODES:
            fname = FUN_TEMPLATE.format(mode=mode, max_evals=max_evals, seed=seed)
            front = load_front(fname)
            if front is None:
                missing.append(fname)
                igd_val = None
                hv_val = None
            else:
                igd_val = igd(reference, front)
                hv_val, hv_status = try_hypervolume(front)
            results.append({
                'seed': seed,
                'mode': mode,
                'file': fname,
                'igd': igd_val,
                'hv': hv_val,
            })

    # write CSV
    out_csv = f'benchmark_{max_evals}_10seeds_metrics.csv'
    with open(out_csv, 'w') as f:
        f.write('seed,mode,file,igd,hv\n')
        for r in results:
            f.write(f"{r['seed']},{r['mode']},{r['file']},{r['igd']},{r['hv']}\n")

    # aggregated stats
    import statistics
    summary = {}
    for mode in MODES:
        igd_vals = [r['igd'] for r in results if r['mode']==mode and r['igd'] is not None]
        hv_vals = [r['hv'] for r in results if r['mode']==mode and r['hv'] is not None]
        summary[mode] = {
            'count': len(igd_vals),
            'igd_mean': statistics.mean(igd_vals) if igd_vals else None,
            'igd_stdev': statistics.stdev(igd_vals) if len(igd_vals)>1 else 0.0,
            'hv_mean': statistics.mean(hv_vals) if hv_vals else None,
            'hv_stdev': statistics.stdev(hv_vals) if len(hv_vals)>1 else 0.0,
        }

    # print report
    print('--- IGD / Hypervolume report for 10 seeds (2000 evals) ---')
    print(f'Reference front: {REF_PATHS[0]}')
    if missing:
        print('\nWARNING: missing or unreadable FUN files:')
        for m in missing:
            print(' -', m)
    print('\nPer-mode summary:')
    for mode, s in summary.items():
        print(f"Mode: {mode}")
        print(f"  IGD: count={s['count']} mean={s['igd_mean']} stdev={s['igd_stdev']}")
        if s['hv_mean'] is not None:
            print(f"  HV:  mean={s['hv_mean']} stdev={s['hv_stdev']}")
        else:
            print('  HV:  not available or failed for all runs')

    print('\nCSV written to', out_csv)

if __name__ == '__main__':
    main()
