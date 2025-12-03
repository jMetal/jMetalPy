# Legacy Files

This directory contains files from the original flat tuning package structure.
They are kept for reference during the transition period.

## Files

| File | Status | New Location |
|------|--------|--------------|
| `observer.py` | Deprecated | `observers/` module (split into base, console, plot, file, rich) |
| `tuning_sequential.py` | Deprecated | `cli/sequential.py` |
| `run_parallel_tuning.py` | Deprecated | `cli/parallel.py` |
| `objective.py` | Deprecated | `metrics/indicators.py` and `metrics/reference_fronts.py` |
| `parameter_space.py` | Deprecated | `algorithms/base.py` (ParameterInfo class) |
| `nsgaii_optuna_tuning.py` | Deprecated | `algorithms/nsgaii.py` (NSGAIITuner class) |
| `nsgaii_validate_tuning.py` | Deprecated | Validation integrated in `algorithms/nsgaii.py` |

## Migration Guide

### Observers
```python
# Old
from jmetal.tuning.observer import TuningProgressObserver

# New
from jmetal.tuning.observers import TuningProgressObserver
# or
from jmetal.tuning import TuningProgressObserver
```

### Metrics
```python
# Old
from jmetal.tuning.objective import compute_indicators, load_reference_front

# New
from jmetal.tuning.metrics import compute_quality_indicators, load_reference_front
```

### CLI
```bash
# Old
python -m jmetal.tuning.tuning_sequential --trials 100

# New
python -m jmetal.tuning.cli.sequential --trials 100
```

### Algorithm Tuning
```python
# Old
from jmetal.tuning.parameter_space import NSGAIIParameters

# New
from jmetal.tuning.algorithms import NSGAIITuner
# Parameters are handled internally by the tuner
```

## Removal Schedule

These files will be removed in a future version once the new modular structure
is fully validated and documented.
