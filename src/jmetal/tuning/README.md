# Algorithm Hyperparameter Tuning with Optuna

This directory contains modular tools for tuning multi-objective algorithm hyperparameters using the Optuna optimization framework.

## Currently Supported Algorithms

- **NSGA-II**: Non-dominated Sorting Genetic Algorithm II

## Quick Start

### Using the High-Level API (Recommended)

```python
from jmetal.tuning import tune

# Simple usage with default problems (ZDT1-6)
result = tune("NSGAII", n_trials=100)
print(f"Best score: {result.best_score}")
print(f"Best params: {result.best_params}")

# Custom problems
from jmetal.problem import ZDT1, ZDT4, DTLZ2
result = tune(
    "NSGAII",
    problems=[ZDT1(), ZDT4(), DTLZ2()],
    n_trials=50,
    sampler="cmaes",
    mode="continuous",
)

# Save results to file
result = tune("NSGAII", n_trials=100, output_path="my_config.json")
```

### Command Line

```bash
# Sequential tuning
python -m jmetal.tuning.tuning_sequential --trials 100

# With options
python -m jmetal.tuning.tuning_sequential --algorithm NSGAII --sampler cmaes --mode continuous
```

## File Structure

```
src/jmetal/tuning/
├── __init__.py            # Package exports
├── tuning.py              # High-level API (tune function)
├── config.py              # Configuration: problems, evaluations, paths
├── algorithms/            # Algorithm-specific tuners
│   ├── __init__.py        # Algorithm registry
│   ├── base.py            # AlgorithmTuner abstract base class
│   └── nsgaii.py          # NSGA-II tuner implementation
├── tuning_sequential.py   # Sequential tuning (no database required)
├── tuning_parallel.py     # Parallel tuning (with PostgreSQL)
├── run_parallel_tuning.sh # Script to launch parallel workers
├── nsgaii_validate_tuning.py  # Configuration validation
└── nsgaii_tuned_config.json   # Tuning results (generated)
```

## Execution Modes

### 1. Sequential Execution (Simple)

No database required. Ideal for quick tests.

```bash
python -m jmetal.tuning.tuning_sequential --trials 100

# Options
python -m jmetal.tuning.tuning_sequential --help
```

**Options:**
- `--algorithm`: Algorithm to tune (default: NSGAII)
- `--trials N`: Number of trials (default: 500)
- `--sampler [tpe|cmaes|random]`: Search algorithm (default: tpe)
- `--mode [categorical|continuous]`: Parameter space mode
- `--seed N`: Seed for reproducibility
- `--output FILE`: Output file for results

### 2. Parallel Execution (PostgreSQL)

Uses PostgreSQL as shared storage to coordinate multiple workers.

**Prerequisites:**
```bash
# Install PostgreSQL (macOS)
brew install postgresql
brew services start postgresql
createdb optuna_jmetal

# Install Python driver
pip install psycopg2-binary
```

**Run:**
```bash
# Use the helper script (4 parallel workers)
./src/jmetal/tuning/run_parallel_tuning.sh 4

# Or manually:
WORKER_ID=0 N_WORKERS=4 python -m jmetal.tuning.tuning_parallel &
WORKER_ID=1 N_WORKERS=4 python -m jmetal.tuning.tuning_parallel &
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

---

## 🔧 Adding Support for New Algorithms

The tuning package uses an extensible architecture based on the Template Method pattern. To add support for a new algorithm (e.g., IBEA, MOEA/D), follow these steps:

### Step 1: Create a New Tuner Class

Create a new file in `algorithms/` (e.g., `ibea.py`):

```python
"""IBEA tuner implementation."""

from dataclasses import dataclass
from typing import Any, Dict

from jmetal.algorithm.multiobjective.ibea import IBEA
from jmetal.core.problem import Problem
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

from .base import AlgorithmTuner


@dataclass
class IBEAConfig:
    """IBEA-specific configuration."""
    kappa: float = 0.05
    population_size: int = 100


class IBEATuner(AlgorithmTuner):
    """Tuner for IBEA algorithm."""
    
    def __init__(
        self,
        population_size: int = 100,
        kappa: float = 0.05,
        **kwargs
    ):
        super().__init__(population_size=population_size, **kwargs)
        self.kappa = kappa
    
    @property
    def name(self) -> str:
        return "IBEA"
    
    def sample_parameters(self, trial, mode: str = "categorical") -> Dict[str, Any]:
        """Sample IBEA hyperparameters."""
        params = {}
        
        # IBEA-specific: kappa parameter
        params["kappa"] = trial.suggest_float("kappa", 0.01, 0.1)
        
        # Crossover
        params["crossover_probability"] = trial.suggest_float(
            "crossover_probability", 0.7, 1.0
        )
        params["crossover_eta"] = trial.suggest_float("crossover_eta", 5.0, 50.0)
        
        # Mutation
        params["mutation_probability_factor"] = trial.suggest_float(
            "mutation_probability_factor", 0.5, 2.0
        )
        params["mutation_eta"] = trial.suggest_float("mutation_eta", 5.0, 50.0)
        
        return params
    
    def create_algorithm(
        self,
        problem: Problem,
        params: Dict[str, Any],
        max_evaluations: int,
    ) -> IBEA:
        """Create configured IBEA instance."""
        n_vars = problem.number_of_variables()
        
        crossover = SBXCrossover(
            probability=params["crossover_probability"],
            distribution_index=params["crossover_eta"],
        )
        
        mutation = PolynomialMutation(
            probability=params["mutation_probability_factor"] / n_vars,
            distribution_index=params["mutation_eta"],
        )
        
        return IBEA(
            problem=problem,
            kappa=params.get("kappa", self.kappa),
            population_size=self.population_size,
            offspring_population_size=self.population_size,
            mutation=mutation,
            crossover=crossover,
            termination_criterion=StoppingByEvaluations(max_evaluations),
        )
```

### Step 2: Register the Tuner

Update `algorithms/__init__.py`:

```python
from .base import AlgorithmTuner, TuningResult
from .nsgaii import NSGAIITuner
from .ibea import IBEATuner  # Add import

# Registry of available tuners
TUNERS = {
    "NSGAII": NSGAIITuner,
    "IBEA": IBEATuner,  # Add to registry
}

__all__ = [
    "AlgorithmTuner",
    "TuningResult", 
    "NSGAIITuner",
    "IBEATuner",  # Add to exports
    "TUNERS",
]
```

### Step 3: Use the New Tuner

```python
from jmetal.tuning import tune

# Now you can tune IBEA!
result = tune("IBEA", n_trials=100)
```

### Architecture Overview

```
AlgorithmTuner (ABC)
├── name: str (abstract property)
├── sample_parameters(trial, mode) -> Dict (abstract)
├── create_algorithm(problem, params, max_evals) -> Algorithm (abstract)
├── evaluate(problem, params) -> Tuple[float, float] (concrete)
├── evaluate_on_problems(problems, params) -> float (concrete)
└── format_params(params) -> str (concrete)

NSGAIITuner(AlgorithmTuner)
└── Implements: name, sample_parameters, create_algorithm

IBEATuner(AlgorithmTuner)  # Example
└── Implements: name, sample_parameters, create_algorithm
```

### Key Methods to Implement

1. **`name`** (property): Return the algorithm name (used in registry and logs)

2. **`sample_parameters(trial, mode)`**: Define the hyperparameter search space
   - Use `trial.suggest_categorical()` for discrete choices
   - Use `trial.suggest_float()` for continuous parameters
   - The `mode` parameter allows different spaces for TPE vs CMA-ES

3. **`create_algorithm(problem, params, max_evaluations)`**: Build the algorithm
   - Receives sampled parameters
   - Must return a configured algorithm instance ready to `run()`

---

## Custom Configuration

Edit `config.py` to change training problems:

```python
# Format: (Problem instance, reference_front_filename)
# The reference_front_filename is the full name of the reference front file 
# (including extension) in the REFERENCE_FRONTS_DIR directory.

TRAINING_PROBLEMS = [
    (ZDT1(), "ZDT1.pf"),                    # Standard ZDT1 reference front
    (ZDT2(), "ZDT2.pf"),                    # Standard ZDT2 reference front
    (MyCustomProblem(), "custom_front.txt"), # Custom reference front file
    # ...
]

# Directory containing reference front files
REFERENCE_FRONTS_DIR = ROOT_DIR / "resources" / "reference_fronts"

TRAINING_EVALUATIONS = 10000
NUMBER_OF_TRIALS = 500
```

When using the `tune()` API directly, you can specify problems in several formats:

```python
from jmetal.tuning import tune
from jmetal.problem import ZDT1, ZDT4

# 1. Just Problem instance (reference front file = problem.name() + ".pf")
result = tune("NSGAII", problems=[ZDT1(), ZDT4()])

# 2. (Problem, reference_front_file) tuple (explicit reference front file)
result = tune("NSGAII", problems=[
    (ZDT1(), "ZDT1.pf"),
    (MyProblem(), "custom_reference.txt"),  # Use custom reference front file
])
```

## Legacy Files

- `nsgaii_optuna_tuning.py`: Original all-in-one script (kept for compatibility)
- `parameter_space.py`: Legacy parameter definitions
- `objective.py`: Legacy objective function
