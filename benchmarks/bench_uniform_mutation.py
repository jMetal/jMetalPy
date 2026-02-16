"""Benchmark UniformMutation: vectorized vs scalar reference.

Run from the repo root with the project's python environment:

    python benchmarks/bench_uniform_mutation.py

The script prints average times (s) for each method and size.
"""
import time
import numpy as np
import os, sys

# Ensure project `src` is on sys.path when running this script directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from jmetal.core.solution import FloatSolution
from jmetal.operator.mutation import UniformMutation
from jmetal.operator.repair import ensure_float_repair


def make_solution_vars(n, lower=0.0, upper=1.0):
    lbs = [lower] * n
    ubs = [upper] * n
    s = FloatSolution(lbs, ubs, 1)
    s._variables = list(np.linspace(lower + 0.1, upper - 0.1, n))
    return s


def scalar_uniform_mutation(vars_list, lbs, ubs, probability, perturbation, rng, repair):
    # scalar reference implementation using repair.repair_scalar
    out = list(vars_list)
    n = len(out)
    for i in range(n):
        if rng.random() < probability:
            var_range = ubs[i] - lbs[i]
            delta = (rng.random() - 0.5) * perturbation * var_range
            new_value = out[i] + delta
            new_value = repair.repair_scalar(new_value, lbs[i], ubs[i])
            out[i] = new_value
    return out


def bench_once(n, probability, perturbation, reps=50):
    s = make_solution_vars(n)
    lbs = s.lower_bound
    ubs = s.upper_bound

    # repair operator and RNGs
    repair = ensure_float_repair(None)

    # vectorized operator uses its own rng (np.random.Generator)
    seed = 12345
    rng_vec = np.random.default_rng(seed)
    op = UniformMutation(probability=probability, perturbation=perturbation, repair_operator=None, rng=rng_vec)

    # measure vectorized
    t0 = time.perf_counter()
    for _ in range(reps):
        s_v = make_solution_vars(n)
        op.execute(s_v)
    t_vec = time.perf_counter() - t0

    # scalar: use RNG with same seed to produce comparable random draws
    rng_scalar = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for _ in range(reps):
        s_s = make_solution_vars(n)
        _ = scalar_uniform_mutation(s_s._variables, s_s.lower_bound, s_s.upper_bound, probability, perturbation, rng_scalar, repair)
    t_scalar = time.perf_counter() - t0

    return t_scalar, t_vec


def main():
    sizes = [100, 1000, 5000, 20000]
    reps_map = {100: 200, 1000: 100, 5000: 40, 20000: 10}
    probability = 0.5
    perturbation = 0.5

    print("UniformMutation benchmark (scalar vs vectorized)")
    print("size\treps\tscalar(s)\tvector(s)\tscalar/vec")
    for n in sizes:
        reps = reps_map.get(n, 20)
        t_scalar, t_vec = bench_once(n, probability, perturbation, reps)
        print(f"{n}\t{reps}\t{t_scalar:.4f}\t{t_vec:.4f}\t{t_scalar/t_vec:.2f}")

if __name__ == '__main__':
    main()
