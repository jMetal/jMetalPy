# General idea

We want to design a jMetalPy package to find the best configuration of a multi-objective metaheuristic (e.g., NSGA-II) for a set of problems (the training set). Optuna will be used as the automated configuration tool.

# Objective function components

The objective function optimized by Optuna must include:

# Optuna tuning protocol for jMetalPy

This document specifies a recommended protocol for using Optuna to tune multi-objective metaheuristics (for example, NSGA-II) over a set of benchmark problems. It defines required inputs, a deterministic evaluation protocol (including repetitions and seeding), the aggregation rule to produce a scalar objective, and operational recommendations for parallelism, reproducibility and artifact handling.

## 1. Scope and goals

This specification covers the experimental protocol used to automatically search for the best configuration of a multi-objective metaheuristic over a training set of problems using Optuna. The goal is to produce a single scalar score per configuration that Optuna minimizes.

It does not prescribe algorithm internals beyond the interface: each evaluation must return the solution front(s) for each problem so quality indicators can be computed.

## 2. Terminology and conventions

- Problem: an optimization problem instance (e.g., `ZDT1`).
- Training set: a collection of problems used to evaluate a configuration.
- Reference front: a high-quality Pareto approximation for a problem (array of objective vectors).
- Indicator: quality indicators such as Normalized Hypervolume (NHV), Inverted Generational Distance (IGD), Additive Epsilon (EP / Epsilon). (Use NHV instead of raw HV; NHV is bounded in [0, 1] and is a minimization indicator.)
- Convention: the protocol treats the overall score as a minimization objective. Indicators used in aggregation must be lower-is-better; prefer NHV (which already follows this convention) and other indicators that have optimum at 0.0.

## 3. Objective function components (required inputs)

The objective function must accept or be configured with:
1. A list of problems composing the training set (P).
2. Reference fronts for each problem in P.
3. A set of quality indicators I to compute for each result front (e.g., NHV, IGD+, EP).
4. The target metaheuristic and its parameter search space S.
5. Fixed algorithm parameters (population size, maximum number of evaluations) unless they are part of S.
6. Repetition count N (default: 1) per (algorithm, problem) pair.

## 4. Evaluation protocol

This section defines the steps executed for each Optuna trial (a sampled configuration) and the aggregation used to produce a scalar score.

### 4.1 High-level steps per trial

1. Sample a configuration `cfg` from the parameter search space `S`.
2. For each problem `p` in the training set `P`:
	a. Execute the algorithm `cfg` on `p` for `r = 1..N` independent runs.
	b. Collect the resulting solution front `F_{p,r}` for each run.
3. For each `p` and each indicator `i` in `I`, compute `value_{p,i,r} = indicator_i(F_{p,r})`.
4. Aggregate per-run values to compute a per-problem, per-indicator mean:

	mean_{p,i} = (1/N) * sum_{r=1..N} value_{p,i,r}

5. Compute the overall score returned to Optuna by summing the per-problem, per-indicator means:

	overall_score = sum_{p in P} sum_{i in I} mean_{p,i}

	(This formula yields a scalar to minimize.)

Notes:
- Default `N = 1` to keep tuning affordable. Increase `N` (e.g. to 3 or 5) to reduce variance in final comparisons.
- Prefer NHV over raw HV: NHV is bounded to [0,1] and is lower-is-better, so when using NHV together with indicators like IGD, IGD+ and Additive Epsilon (which all have optimal value 0.0) explicit normalization is not required. If a different higher-is-better indicator is included, transform it before aggregation.

### 4.2 Seeding and reproducibility

- For reproducibility each run must receive a deterministic seed derived from a base seed, the trial id (or trial number), and the run index. Example deterministic formula:

  seed = base_seed + trial_number * 1000 + run_index

- The seed used for each run must be recorded and saved with trial metadata.

### 4.3 Normalization and scaling

- When the chosen indicator set includes `NHV` and indicators that by convention are optimal at `0.0` (for example IGD, IGD+, Additive Epsilon), explicit normalization is generally not required because all values are comparable in the sense that lower-is-better and NHV ∈ [0,1].
- If you include indicators that are not on a compatible scale (e.g., other higher-is-better measures), either remove them or apply a deterministic transformation so that they become lower-is-better and roughly comparable; document any transformation used.

### 4.4 Handling degenerate cases

- If the hypervolume of the reference front is zero (HV_ref == 0) the protocol must treat the problem as invalid for HV-based indicators and **fail early by raising a clear error**. A zero HV makes normalization impossible (division by zero) and therefore the evaluation should not continue for that problem. Implementations must raise a descriptive exception so the caller (or the tuning harness) can log the failure and either abort the trial or skip the configuration according to the experimental policy.

## 5. Parallelism and storage

- Optuna's storage backends behave differently under concurrent writes. If using SQLite as the storage URL, prefer `n_jobs=1` (single process) to avoid SQLite locking issues. For parallel trials use a client-server DB (Postgres) or Optuna's RDB backend that supports concurrency.
- Alternatively, run Optuna with `n_jobs=1` and parallelize the internal evaluation across problems or runs where safe.

## 6. Reproducibility and artifact logging

For each trial, the system should persist the following artifacts and metadata:
- Trial id and full parameter configuration.
- Per-run random seeds and run indices.
- Per-problem per-run solution fronts (preferably `FUN`/`VAR` files) or serialized arrays.
- Per-problem per-run indicator values.
- Computed aggregated values (`mean_{p,i}`) and the `overall_score` returned to Optuna.
- Location/URI of any saved files.

This metadata enables post-hoc analysis and exact re-execution of promising configurations.

## 7. Practical recommendations

- Start tuning with `N = 1` and small per-run budgets (reduced population and/or evaluations) to perform quick exploratory tuning.
- After a set of promising configurations is found, re-evaluate the top-K configurations with larger `N` (e.g., 3–5) and full budgets to obtain robust final comparisons.
- For production-scale parallel tuning, use a server database (Postgres) as Optuna storage.

## 8. Output format and artifacts

- Recommended artifacts per (trial, run, problem): `FUN.<trial>.<problem>.<run>.csv`, `VAR.<trial>.<problem>.<run>.csv` plus a JSON summary with indicators and seeds.

## 9. Minimal example workflow (pseudocode)

```
base_seed = 1234
for trial in optuna.trials:
	 cfg = sample_configuration(trial)
	 trial_number = trial.number
	 all_values = []
	 for p in problems:
		  for r in range(N):
				seed = base_seed + trial_number * 1000 + r
				front = run_algorithm(cfg, problem=p, seed=seed)
				for i in indicators:
					 value = compute_indicator(i, front, reference_front[p])
					 record(value, trial_number, p, r, i, seed)
	 aggregate means and compute overall_score
	 return overall_score
```

## 10. Appendix: implementation notes

- When deriving a reference point from a reference front use element-wise maxima and apply a small positive offset (e.g., `reference_point_offset`) to ensure the reference point is strictly worse than frontier extremes. This is already implemented in the NHV helper used by the prototype.
- Document the exact normalization strategy used for indicators in any experiment report.
