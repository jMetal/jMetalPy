# Component-Based Architecture

`jmetal.component` assembles algorithms from small, independently testable components instead of
defining them by subclassing. It mirrors jMetal Java's `jmetal-component` module and lives
alongside `jmetal.algorithm`, which keeps working unchanged -- nothing here replaces the classic
algorithms, and no existing example, notebook, or user code is affected.

This page currently covers evolutionary algorithms (EA/MOEA). A PSO template and catalogue are
planned but not implemented yet.

## Why

The classic algorithms hard-wire their steps as methods on a class hierarchy: `GeneticAlgorithm`
defines `create_initial_solutions()`, `evaluate()`, `selection()`, `reproduction()`, and
`NSGAII.replacement()` overrides the base class's replacement step. Swapping one step for another
means subclassing. A component-based algorithm instead takes six collaborators as constructor
arguments, so any of them can be swapped independently without touching the algorithm's control
flow at all.

## The six components

| Component | Responsibility | Protocol method(s) |
|---|---|---|
| `SolutionsCreation` | Creates the initial population | `create() -> list[S]` |
| `Evaluation` | Evaluates a population against a problem | `evaluate(list[S]) -> list[S]`, `computed_evaluations() -> int` |
| `Termination` | Decides when to stop | `is_met(status: Mapping) -> bool` |
| `Selection` | Builds a mating pool from the population | `select(list[S]) -> list[S]` |
| `Variation` | Turns a mating pool into offspring | `variate(...) -> list[S]`, `mating_pool_size() -> int`, `offspring_population_size() -> int` |
| `Replacement` | Selects survivors from parents and offspring | `replace(current, offspring) -> list[S]` |

Each is a [`typing.Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)
(except `Replacement`, an `ABC` -- see below):
structural typing means any object exposing the right method already satisfies the contract, with
no need to inherit from anything. A bare function wrapped in a small object, a `lambda`, or an
existing class all work.

`Replacement` is the one exception: the three replacement strategies that already existed in
`jmetal.operator.replacement` (`RankingAndDensityEstimatorReplacement`,
`RankingAndCrowdingDistanceReplacement`, `SMSEMOAReplacement`) only shared their `replace()` method
by convention, with no common base to type against. `jmetal.operator.replacement.Replacement` is a
minimal `ABC` that fixes that, and `jmetal.component.catalogue.ea.replacement` re-exports it
directly -- no adapter needed, since the existing classes already match the shape the component
model expects.

## The `EvolutionaryAlgorithm` template

`jmetal.component.algorithm.evolutionary_algorithm.EvolutionaryAlgorithm` is a direct translation
of jMetal Java's `EvolutionaryAlgorithm` template. Its `run()` is exactly:

```python
population = solutions_creation.create()
population = evaluation.evaluate(population)
while not termination.is_met(state):
    mating_population = selection.select(population)
    offspring_population = variation.variate(population, mating_population)
    offspring_population = evaluation.evaluate(offspring_population)
    population = replacement.replace(population, offspring_population)
```

Unlike `jmetal.core.algorithm.Algorithm`, it does not inherit from `threading.Thread`: nothing in
jMetalPy calls `start()`/`join()` on an algorithm, so the template skips that coupling instead of
carrying it for no benefit.

## Building NSGA-II

`build_nsgaii()` is a factory *function*, not a chainable builder class. jMetal Java's
`NSGAIIBuilder` uses `.setX().build()` because Java has no keyword arguments; Python already does,
so a function with keyword-only overrides gives the same flexibility without an intermediate
mutable object:

```python
from jmetal.component.algorithm.multiobjective.nsgaii import build_nsgaii
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT1

problem = ZDT1()
algorithm = build_nsgaii(
    problem,
    population_size=100,
    offspring_population_size=100,
    crossover=SBXCrossover(probability=1.0, distribution_index=20),
    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
)

algorithm.run()
front = algorithm.result()
```

Every default -- `selection`, `variation`, `replacement`, `termination` -- can be overridden with a
keyword argument, and every existing observer in `jmetal.util.observer`
(`ProgressBarObserver`, `VisualizerObserver`, `WriteFrontToFileObserver`, ...) works against
`algorithm.observable` unchanged, since `AlgorithmState.as_dict()` produces the same
`"PROBLEM"`/`"EVALUATIONS"`/`"SOLUTIONS"`/`"COMPUTING_TIME"`-keyed mapping the classic algorithms
already notify with.

## Reproducibility

No classic jMetalPy algorithm accepts a seed. `EvolutionaryAlgorithm` accepts an optional
`rng: np.random.Generator`, shared with any component that already declares a plain `rng`
attribute:

```python
import numpy as np

algorithm = build_nsgaii(..., rng=np.random.default_rng(42))
```

Not every operator is RNG-aware yet -- `SBXCrossover` and `PolynomialMutation` already accept an
explicit `rng`, but `TournamentSelection` and `Problem.create_solution()` still draw from Python's
global `random` module. Component-based algorithms are reproducible today when built with
RNG-aware operators and a reseeded global `random` state; migrating the remaining operators to the
injectable-`rng` pattern is ongoing, tracked in `MODERNIZATION.md`.

## Verified behavioral equivalence

`build_nsgaii(...)` and the classic `NSGAII(...)` produce **identical** final populations on ZDT1
and DTLZ2 given the same seed -- both reuse the exact same operator classes and
`Problem.create_solution()`, so identical random state produces identical results. This is checked
by a dedicated test (`tests/component/algorithm/multiobjective/test_nsgaii_equivalence.py`) and is
the acceptance criterion for this architecture: adopting components changes *how* an algorithm is
assembled, not *what* it computes.

## Package layout

```text
src/jmetal/component/
├── algorithm/
│   ├── algorithm_state.py              # AlgorithmState
│   ├── evolutionary_algorithm.py       # EvolutionaryAlgorithm
│   └── multiobjective/
│       └── nsgaii.py                   # build_nsgaii()
└── catalogue/
    ├── common/
    │   ├── solutions_creation.py
    │   ├── evaluation.py
    │   └── termination.py
    └── ea/
        ├── selection.py
        ├── variation.py
        └── replacement.py
```

This currently covers NSGA-II only. Further MOEAs (SPEA2, SMS-EMOA, MOCell) reusing the same
catalogue, and a PSO template and catalogue, are planned -- see `MODERNIZATION.md`.
