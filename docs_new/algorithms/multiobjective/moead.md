# MOEA/D

[MOEA/D](https://doi.org/10.1109/TEVC.2007.892759) decomposes a multi-objective problem into a set
of scalar subproblems, defined by a set of weight vectors, and solves them simultaneously by
exploiting the neighborhood relationships between subproblems.

```python
--8<-- "examples/multiobjective/moead/moead_dtlz1.py"
```

See `examples/multiobjective/moead/` for the MOEA/D-DRA and MOEA/D-IEpsilon variants.
