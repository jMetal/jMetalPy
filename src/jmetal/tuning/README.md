# jMetal Hyperparameter Tuning Package

This package provides tools for hyperparameter tuning of multi-objective optimization algorithms using [Optuna](https://optuna.org/).

## Package Structure

```
tuning/
├── __init__.py          # Public API exports
├── tuning.py            # High-level API (tune, describe_parameters)
├── tuning_parallel.py   # Parallel worker implementation
├── algorithms/          # Algorithm-specific tuners
│   ├── base.py          # AlgorithmTuner ABC, TuningResult, ParameterInfo
│   └── nsgaii.py        # NSGAIITuner implementation
├── observers/           # Progress visualization
│   ├── base.py          # TuningObserver ABC
│   ├── console.py       # TuningProgressObserver
│   ├── plot.py          # TuningPlotObserver
│   ├── file.py          # TuningFileObserver
│   └── rich.py          # TuningRichObserver
├── metrics/             # Quality indicators
│   ├── indicators.py    # compute_quality_indicators, aggregate_scores
│   └── reference_fronts.py  # load_reference_front, get_reference_point
├── config/              # Configuration
│   ├── defaults.py      # Tuning parameters (evaluations, trials, seed)
│   ├── paths.py         # File and directory paths
│   └── problems.py      # Training problem definitions
├── cli/                 # Command-line interfaces
│   ├── sequential.py    # Single-process tuning
│   └── parallel.py      # Multi-worker tuning
├── runners/             # (Future) Execution runners
└── _legacy/             # Deprecated files for reference
```

## Quick Start

### High-Level API

```python
from jmetal.tuning import tune

# Simple tuning
result = tune("NSGAII", n_trials=100)
print(result.best_params)

# With progress observers
from jmetal.tuning import TuningProgressObserver, TuningPlotObserver

result = tune(
    "NSGAII",
    n_trials=100,
    observers=[TuningProgressObserver(), TuningPlotObserver()]
)
```

### Describe Parameters

```python
from jmetal.tuning import describe_parameters

# View tunable parameters
print(describe_parameters("NSGAII"))

# Export to JSON
describe_parameters("NSGAII", format="json", output_path="params.json")
```

### Command-Line Interface

```bash
# Sequential tuning (in-memory)
python -m jmetal.tuning.cli.sequential --trials 100

# Parallel tuning (requires PostgreSQL)
python -m jmetal.tuning.cli.parallel --workers 4 --trials 500
```

## Observers

Progress observers provide real-time feedback during tuning:

| Observer | Description |
|----------|-------------|
| `TuningProgressObserver` | Console progress bar with statistics |
| `TuningPlotObserver` | Live matplotlib plot of best values |
| `TuningFileObserver` | CSV logging of trial results |
| `TuningRichObserver` | Enhanced console output (requires rich) |

### Custom Observers

```python
from jmetal.tuning.observers import TuningObserver

class MyObserver(TuningObserver):
    def on_trial_complete(self, study, trial):
        # Custom logic
        pass
```

## Configuration

Default configuration can be modified via `jmetal.tuning.config`:

```python
from jmetal.tuning.config import (
    POPULATION_SIZE,      # 100
    TRAINING_EVALUATIONS, # 10000
    NUMBER_OF_TRIALS,     # 500
    TRAINING_PROBLEMS,    # ZDT1-ZDT6
)
```

### Custom Problem Sets

```python
from jmetal.tuning.config.problems import create_problem_set
from jmetal.problem import DTLZ1, DTLZ2

problems = create_problem_set([DTLZ1, DTLZ2])
```

## Algorithm Tuners

### Built-in Tuners

- `NSGAIITuner`: Tunes crossover, mutation, and selection parameters

### Creating Custom Tuners

```python
from jmetal.tuning.algorithms import AlgorithmTuner, ParameterInfo

class MyAlgorithmTuner(AlgorithmTuner):
    @property
    def algorithm_name(self) -> str:
        return "MyAlgorithm"
    
    @property
    def parameters(self) -> List[ParameterInfo]:
        return [
            ParameterInfo("param1", "float", (0.0, 1.0), 0.5, "Description"),
        ]
    
    def create_algorithm(self, params, problem):
        # Create and return algorithm instance
        pass
```

## Metrics

Quality indicators for evaluating Pareto fronts:

```python
from jmetal.tuning.metrics import (
    compute_quality_indicators,
    load_reference_front,
    aggregate_scores,
)

# Load reference front
ref_front = load_reference_front("ZDT1.pf")

# Compute indicators
nhv, ae = compute_quality_indicators(solutions, ref_front)
```

## NSGA-II Parameter Space

| Parameter | Range | Description |
|-----------|-------|-------------|
| `offspring_population_size` | [1, 10, 50, 100, 150, 200] | Offspring population size |
| `crossover_type` | [sbx, blxalpha] | Crossover operator type |
| `crossover_probability` | [0.7, 1.0] | Crossover probability |
| `crossover_eta` | [5, 400] | SBX distribution index |
| `blx_alpha` | [0, 1] | BLX alpha parameter |
| `mutation_type` | [polynomial, uniform] | Mutation operator type |
| `mutation_probability_factor` | [0.5, 2.0] | Probability factor (prob = factor/n) |
| `mutation_eta` | [5, 400] | Polynomial distribution index |
| `mutation_perturbation` | [0.1, 2.0] | Uniform mutation perturbation |

## Adding Support for New Algorithms

The tuning package uses an extensible architecture. To add a new algorithm:

1. **Create a tuner class** in `algorithms/` implementing `AlgorithmTuner`
2. **Register it** in `algorithms/__init__.py` TUNERS dictionary
3. **Use it** via `tune("YourAlgorithm", n_trials=100)`

See `algorithms/nsgaii.py` for a complete example.

## Dependencies

- **Required**: `optuna>=4.0`
- **Optional**: 
  - `psycopg2-binary` for parallel tuning with PostgreSQL
  - `matplotlib` for TuningPlotObserver
  - `rich` for TuningRichObserver

## Legacy Files

Deprecated files are in `_legacy/` directory. See `_legacy/README.md` for migration guide.

## License

Part of jMetalPy - MIT License
