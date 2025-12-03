"""
Configuration classes for hyperparameter tuning.

This module provides dataclasses for configuring tuning experiments,
with support for loading from YAML files.

Example YAML configuration:
    algorithm: NSGAII
    n_trials: 100
    n_evaluations: 10000
    seed: 42
    
    parameter_space:
      crossover:
        type: sbx
        probability: {min: 0.8, max: 1.0}
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from jmetal.core.problem import Problem


@dataclass
class ParameterRange:
    """
    Defines a range for a continuous parameter.
    
    Attributes:
        min: Minimum value
        max: Maximum value
    """
    min: float
    max: float
    
    def __post_init__(self):
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")


@dataclass
class CategoricalParameter:
    """
    Defines choices for a categorical parameter.
    
    Attributes:
        values: List of possible values
    """
    values: List[Any]
    
    def __post_init__(self):
        if not self.values:
            raise ValueError("Categorical parameter must have at least one value")


@dataclass
class CrossoverConfig:
    """
    Configuration for crossover operator(s).
    
    Attributes:
        types: List of crossover types to consider ["sbx", "blxalpha"]
        probability: Range or fixed value for crossover probability
        sbx_distribution_index: Range for SBX eta parameter
        blx_alpha: Range for BLX-alpha parameter
    """
    types: List[str] = field(default_factory=lambda: ["sbx", "blxalpha"])
    probability: Union[float, ParameterRange] = field(
        default_factory=lambda: ParameterRange(0.7, 1.0)
    )
    sbx_distribution_index: Union[float, ParameterRange] = field(
        default_factory=lambda: ParameterRange(5.0, 400.0)
    )
    blx_alpha: Union[float, ParameterRange] = field(
        default_factory=lambda: ParameterRange(0.0, 1.0)
    )
    
    def is_fixed_type(self) -> bool:
        """Check if only one crossover type is specified."""
        return len(self.types) == 1
    
    def get_fixed_type(self) -> Optional[str]:
        """Get the fixed type if only one is specified."""
        return self.types[0] if self.is_fixed_type() else None


@dataclass
class MutationConfig:
    """
    Configuration for mutation operator(s).
    
    Attributes:
        types: List of mutation types to consider ["polynomial", "uniform"]
        probability_factor: Range for mutation probability factor (prob = factor/n_vars)
        polynomial_distribution_index: Range for polynomial mutation eta
        uniform_perturbation: Range for uniform mutation perturbation
    """
    types: List[str] = field(default_factory=lambda: ["polynomial", "uniform"])
    probability_factor: Union[float, ParameterRange] = field(
        default_factory=lambda: ParameterRange(0.5, 2.0)
    )
    polynomial_distribution_index: Union[float, ParameterRange] = field(
        default_factory=lambda: ParameterRange(5.0, 400.0)
    )
    uniform_perturbation: Union[float, ParameterRange] = field(
        default_factory=lambda: ParameterRange(0.1, 2.0)
    )
    
    def is_fixed_type(self) -> bool:
        """Check if only one mutation type is specified."""
        return len(self.types) == 1
    
    def get_fixed_type(self) -> Optional[str]:
        """Get the fixed type if only one is specified."""
        return self.types[0] if self.is_fixed_type() else None


@dataclass  
class ParameterSpaceConfig:
    """
    Configuration for the hyperparameter search space.
    
    Attributes:
        offspring_population_size: Categorical values or range for offspring size
        crossover: Crossover operator configuration
        mutation: Mutation operator configuration
    """
    offspring_population_size: Union[CategoricalParameter, List[int]] = field(
        default_factory=lambda: CategoricalParameter([1, 10, 50, 100, 150, 200])
    )
    crossover: CrossoverConfig = field(default_factory=CrossoverConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)


@dataclass
class ProblemConfig:
    """
    Configuration for a training problem.
    
    Attributes:
        name: Problem class name (e.g., "ZDT1", "DTLZ2")
        reference_front: Reference front filename (e.g., "ZDT1.pf")
        kwargs: Additional arguments for problem constructor
    """
    name: str
    reference_front: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig:
    """
    Configuration for tuning output.
    
    Attributes:
        path: Path for saving best configuration JSON
        save_history: Whether to save full optimization history
        history_path: Path for history file (if save_history is True)
    """
    path: str = "./nsgaii_tuned_config.json"
    save_history: bool = False
    history_path: str = "./tuning_history.csv"


@dataclass
class TuningConfig:
    """
    Complete configuration for a hyperparameter tuning experiment.
    
    This class encapsulates all settings needed for tuning and can be
    loaded from a YAML file or constructed programmatically.
    
    Attributes:
        algorithm: Algorithm to tune (e.g., "NSGAII")
        n_trials: Number of Optuna trials
        n_evaluations: Max function evaluations per problem per trial
        population_size: Fixed population size for the algorithm
        seed: Random seed for reproducibility
        sampler: Optuna sampler ("tpe", "cmaes", "random")
        parameter_space: Search space configuration
        problems: List of training problems
        output: Output configuration
        
    Example:
        # Load from YAML
        config = TuningConfig.from_yaml("tuning_config.yaml")
        
        # Use with tune()
        result = tune(config=config)
        
        # Or construct programmatically
        config = TuningConfig(
            n_trials=50,
            n_evaluations=5000,
        )
    """
    algorithm: str = "NSGAII"
    n_trials: int = 100
    n_evaluations: int = 10000
    population_size: int = 100
    seed: int = 42
    sampler: str = "tpe"
    
    parameter_space: ParameterSpaceConfig = field(default_factory=ParameterSpaceConfig)
    problems: List[ProblemConfig] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def __post_init__(self):
        """Set default problems if none specified."""
        if not self.problems:
            self.problems = [
                ProblemConfig("ZDT1", "ZDT1.pf"),
                ProblemConfig("ZDT2", "ZDT2.pf"),
                ProblemConfig("ZDT3", "ZDT3.pf"),
                ProblemConfig("ZDT4", "ZDT4.pf"),
                ProblemConfig("ZDT6", "ZDT6.pf"),
            ]
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TuningConfig":
        """
        Load configuration from a YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            TuningConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TuningConfig":
        """
        Create configuration from a dictionary.
        
        Args:
            data: Dictionary with configuration values
            
        Returns:
            TuningConfig instance
        """
        # Parse parameter space
        param_space = None
        if "parameter_space" in data:
            param_space = _parse_parameter_space(data["parameter_space"])
        
        # Parse problems
        problems = []
        if "problems" in data:
            for p in data["problems"]:
                if isinstance(p, dict):
                    problems.append(ProblemConfig(
                        name=p["name"],
                        reference_front=p["reference_front"],
                        kwargs=p.get("kwargs", {}),
                    ))
                else:
                    # Simple format: just problem name
                    problems.append(ProblemConfig(p, f"{p}.pf"))
        
        # Parse output
        output = OutputConfig()
        if "output" in data:
            out_data = data["output"]
            output = OutputConfig(
                path=out_data.get("path", output.path),
                save_history=out_data.get("save_history", output.save_history),
                history_path=out_data.get("history_path", output.history_path),
            )
        
        return cls(
            algorithm=data.get("algorithm", "NSGAII"),
            n_trials=data.get("n_trials", 100),
            n_evaluations=data.get("n_evaluations", 10000),
            population_size=data.get("population_size", 100),
            seed=data.get("seed", 42),
            sampler=data.get("sampler", "tpe"),
            parameter_space=param_space or ParameterSpaceConfig(),
            problems=problems if problems else None,  # None triggers default
            output=output,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary (for YAML export).
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "algorithm": self.algorithm,
            "n_trials": self.n_trials,
            "n_evaluations": self.n_evaluations,
            "population_size": self.population_size,
            "seed": self.seed,
            "sampler": self.sampler,
            "parameter_space": _parameter_space_to_dict(self.parameter_space),
            "problems": [
                {"name": p.name, "reference_front": p.reference_front, **p.kwargs}
                for p in self.problems
            ],
            "output": {
                "path": self.output.path,
                "save_history": self.output.save_history,
                "history_path": self.output.history_path,
            },
        }
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            path: Output path for YAML file
        """
        path = Path(path)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def get_problems_as_tuples(self) -> List[Tuple[Problem, str]]:
        """
        Convert problem configs to (Problem, reference_front) tuples.
        
        Returns:
            List of (Problem instance, reference_front_filename) tuples
        """
        from jmetal import problem as jmetal_problems
        
        result = []
        for p in self.problems:
            # Get problem class from jmetal.problem module
            problem_class = getattr(jmetal_problems, p.name, None)
            if problem_class is None:
                raise ValueError(f"Unknown problem: {p.name}")
            
            # Instantiate with kwargs
            problem_instance = problem_class(**p.kwargs)
            result.append((problem_instance, p.reference_front))
        
        return result


def _parse_parameter_space(data: Dict[str, Any]) -> ParameterSpaceConfig:
    """Parse parameter_space section from YAML data."""
    config = ParameterSpaceConfig()
    
    # Offspring population size
    if "offspring_population_size" in data:
        ops_data = data["offspring_population_size"]
        if isinstance(ops_data, dict):
            if "values" in ops_data:
                config.offspring_population_size = CategoricalParameter(ops_data["values"])
        elif isinstance(ops_data, list):
            config.offspring_population_size = CategoricalParameter(ops_data)
        elif isinstance(ops_data, int):
            config.offspring_population_size = CategoricalParameter([ops_data])
    
    # Crossover
    if "crossover" in data:
        config.crossover = _parse_crossover_config(data["crossover"])
    
    # Mutation
    if "mutation" in data:
        config.mutation = _parse_mutation_config(data["mutation"])
    
    return config


def _parse_crossover_config(data: Dict[str, Any]) -> CrossoverConfig:
    """Parse crossover configuration from YAML data."""
    config = CrossoverConfig()
    
    # Types
    if "type" in data:
        # Single type specified
        config.types = [data["type"]]
    elif "types" in data:
        config.types = data["types"]
    
    # Probability
    if "probability" in data:
        config.probability = _parse_value_or_range(data["probability"])
    
    # SBX distribution index
    if "distribution_index" in data:
        config.sbx_distribution_index = _parse_value_or_range(data["distribution_index"])
    elif "sbx" in data and "distribution_index" in data["sbx"]:
        config.sbx_distribution_index = _parse_value_or_range(data["sbx"]["distribution_index"])
    
    # BLX alpha
    if "alpha" in data:
        config.blx_alpha = _parse_value_or_range(data["alpha"])
    elif "blxalpha" in data and "alpha" in data["blxalpha"]:
        config.blx_alpha = _parse_value_or_range(data["blxalpha"]["alpha"])
    
    return config


def _parse_mutation_config(data: Dict[str, Any]) -> MutationConfig:
    """Parse mutation configuration from YAML data."""
    config = MutationConfig()
    
    # Types
    if "type" in data:
        config.types = [data["type"]]
    elif "types" in data:
        config.types = data["types"]
    
    # Probability factor
    if "probability_factor" in data:
        config.probability_factor = _parse_value_or_range(data["probability_factor"])
    
    # Polynomial distribution index
    if "distribution_index" in data:
        config.polynomial_distribution_index = _parse_value_or_range(data["distribution_index"])
    elif "polynomial" in data and "distribution_index" in data["polynomial"]:
        config.polynomial_distribution_index = _parse_value_or_range(
            data["polynomial"]["distribution_index"]
        )
    
    # Uniform perturbation
    if "perturbation" in data:
        config.uniform_perturbation = _parse_value_or_range(data["perturbation"])
    elif "uniform" in data and "perturbation" in data["uniform"]:
        config.uniform_perturbation = _parse_value_or_range(data["uniform"]["perturbation"])
    
    return config


def _parse_value_or_range(data: Any) -> Union[float, ParameterRange]:
    """Parse a value that can be either fixed or a range."""
    if isinstance(data, dict):
        return ParameterRange(min=data["min"], max=data["max"])
    else:
        return float(data)


def _parameter_space_to_dict(config: ParameterSpaceConfig) -> Dict[str, Any]:
    """Convert ParameterSpaceConfig to dictionary for YAML export."""
    def value_or_range_to_dict(v):
        if isinstance(v, ParameterRange):
            return {"min": v.min, "max": v.max}
        return v
    
    # Offspring
    if isinstance(config.offspring_population_size, CategoricalParameter):
        ops = {"values": config.offspring_population_size.values}
    else:
        ops = {"values": config.offspring_population_size}
    
    return {
        "offspring_population_size": ops,
        "crossover": {
            "types": config.crossover.types,
            "probability": value_or_range_to_dict(config.crossover.probability),
            "sbx": {
                "distribution_index": value_or_range_to_dict(config.crossover.sbx_distribution_index),
            },
            "blxalpha": {
                "alpha": value_or_range_to_dict(config.crossover.blx_alpha),
            },
        },
        "mutation": {
            "types": config.mutation.types,
            "probability_factor": value_or_range_to_dict(config.mutation.probability_factor),
            "polynomial": {
                "distribution_index": value_or_range_to_dict(config.mutation.polynomial_distribution_index),
            },
            "uniform": {
                "perturbation": value_or_range_to_dict(config.mutation.uniform_perturbation),
            },
        },
    }
