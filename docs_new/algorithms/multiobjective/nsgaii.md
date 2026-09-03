# NSGA-II

[NSGA-II](https://doi.org/10.1109/4235.996017) (Non-dominated Sorting Genetic Algorithm II) is the
reference multi-objective evolutionary algorithm: it ranks the population using non-dominated
sorting and uses the crowding distance as a density estimator to preserve diversity.

```python
--8<-- "examples/multiobjective/nsgaii/nsgaii_standard_settings.py"
```

`examples/multiobjective/nsgaii/` has many more variants: steady-state, dynamic, distributed
(Dask, Spark, multiprocessing), preference-based, real-time plotting, binary and mixed encodings,
constrained problems, and archive-based variants.
