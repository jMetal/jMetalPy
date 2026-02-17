"""
Base class for algorithm tuners.

This module defines the abstract interface that all algorithm-specific
tuners must implement.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from jmetal.core.problem import Problem
from jmetal.core.quality_indicator import NormalizedHyperVolume, AdditiveEpsilonIndicator
from jmetal.util.solution import get_non_dominated_solutions, read_solutions
from jmetal.tuning.config import POPULATION_SIZE as DEFAULT_POPULATION_SIZE


@dataclass
class ParameterInfo:
    """Description of a tunable parameter."""
    name: str
    type: str  # "float", "int", "categorical"
    description: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None
    conditional_on: Optional[str] = None  # Parameter this depends on
    conditional_value: Optional[Any] = None  # Value that enables this parameter
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d: Dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "description": self.description,
        }
        if self.min_value is not None:
            d["min"] = self.min_value
        if self.max_value is not None:
            d["max"] = self.max_value
        if self.choices is not None:
            d["choices"] = self.choices
        if self.default is not None:
            d["default"] = self.default
        if self.conditional_on is not None:
            d["conditional_on"] = self.conditional_on
            d["conditional_value"] = self.conditional_value
        return d


@dataclass
class TuningResult:
    """Container for tuning results."""
    algorithm_name: str
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    training_problems: List[str]
    training_evaluations: int
    elapsed_seconds: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "algorithm": self.algorithm_name,
            "best_value": self.best_score,
            "best_params": self.best_params,
            "n_trials": self.n_trials,
            "training_problems": self.training_problems,
            "training_evaluations": self.training_evaluations,
            "elapsed_seconds": self.elapsed_seconds,
            **self.extra,
        }


class AlgorithmTuner(ABC):
    """
    Abstract base class for algorithm-specific hyperparameter tuning.
    
    To add support for a new algorithm:
    1. Create a new class that inherits from AlgorithmTuner
    2. Implement the abstract methods: name, sample_parameters, create_algorithm
    3. Optionally override evaluate() if custom evaluation logic is needed
    4. Register the tuner in algorithms/__init__.py
    
    Example:
        class MyAlgorithmTuner(AlgorithmTuner):
            @property
            def name(self) -> str:
                return "MyAlgorithm"
            
            def sample_parameters(self, trial, mode="categorical") -> Dict[str, Any]:
                return {
                    "param1": trial.suggest_float("param1", 0.0, 1.0),
                    "param2": trial.suggest_categorical("param2", ["a", "b"]),
                }
            
            def create_algorithm(self, problem, params, max_evaluations):
                return MyAlgorithm(problem=problem, **params)
    """
    
    def __init__(
        self,
        population_size: int = DEFAULT_POPULATION_SIZE,
        reference_point_offset: float = 0.1,
        reference_fronts_dir: Optional[Path] = None,
    ):
        """
        Initialize the tuner.
        
        Args:
            population_size: Population size for the algorithm
            reference_point_offset: Offset for hypervolume reference point
            reference_fronts_dir: Directory containing reference front files
        """
        self.population_size = population_size
        self.reference_point_offset = reference_point_offset
        
        if reference_fronts_dir is None:
            # Default: jMetalPy/resources/reference_fronts
            self.reference_fronts_dir = (
                Path(__file__).resolve().parent.parent.parent.parent.parent
                / "resources" / "reference_fronts"
            )
        else:
            self.reference_fronts_dir = Path(reference_fronts_dir)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the algorithm name (e.g., 'NSGAII', 'IBEA')."""
        pass
    
    @abstractmethod
    def sample_parameters(self, trial, mode: str = "categorical") -> Dict[str, Any]:
        """
        Sample hyperparameters from an Optuna trial.
        
        Args:
            trial: Optuna trial object
            mode: "categorical" for TPE sampler, "continuous" for CMA-ES
            
        Returns:
            Dictionary of sampled hyperparameters
        """
        pass
    
    @abstractmethod
    def create_algorithm(
        self, 
        problem: Problem, 
        params: Dict[str, Any], 
        max_evaluations: int
    ) -> Any:
        """
        Create an algorithm instance with the given parameters.
        
        Args:
            problem: The optimization problem
            params: Hyperparameters (from sample_parameters)
            max_evaluations: Maximum number of function evaluations
            
        Returns:
            Configured algorithm instance ready to run
        """
        pass
    
    @abstractmethod
    def get_parameter_space(self) -> List[ParameterInfo]:
        """
        Return the description of all tunable parameters.
        
        This method should return a list of ParameterInfo objects describing
        all hyperparameters that can be tuned, including their types, ranges,
        and descriptions suitable for non-expert users.
        
        Returns:
            List of ParameterInfo objects describing the parameter space
        """
        pass
    
    def describe_parameters(self) -> str:
        """
        Generate a human-readable description of the parameter space.
        
        Returns:
            Formatted string describing all tunable parameters
        """
        params = self.get_parameter_space()
        
        lines = [
            f"Parameter Space for {self.name}",
            "=" * (21 + len(self.name)),
            "",
        ]
        
        # Group parameters by whether they're conditional
        main_params = [p for p in params if p.conditional_on is None]
        conditional_params = [p for p in params if p.conditional_on is not None]
        
        # Main parameters
        for param in main_params:
            lines.append(f"• {param.name}")
            lines.append(f"  Description: {param.description}")
            lines.append(f"  Type: {param.type}")
            
            if param.type == "categorical":
                lines.append(f"  Options: {param.choices}")
            else:
                range_str = f"[{param.min_value}, {param.max_value}]"
                lines.append(f"  Range: {range_str}")
            
            if param.default is not None:
                lines.append(f"  Default: {param.default}")
            lines.append("")
        
        # Conditional parameters
        if conditional_params:
            lines.append("Conditional Parameters")
            lines.append("-" * 22)
            lines.append("")
            
            for param in conditional_params:
                lines.append(f"• {param.name}")
                lines.append(f"  Description: {param.description}")
                lines.append(f"  Type: {param.type}")
                
                if param.type == "categorical":
                    lines.append(f"  Options: {param.choices}")
                else:
                    range_str = f"[{param.min_value}, {param.max_value}]"
                    lines.append(f"  Range: {range_str}")
                
                lines.append(f"  Active when: {param.conditional_on} = {param.conditional_value}")
                
                if param.default is not None:
                    lines.append(f"  Default: {param.default}")
                lines.append("")
        
        return "\n".join(lines)
    
    def export_parameter_space(
        self, 
        output_path: Optional[Path] = None,
        format: str = "json"
    ) -> Optional[str]:
        """
        Export the parameter space to a file or return as string.
        
        Args:
            output_path: Path to save the file. If None, returns string.
            format: Output format ("json", "yaml", or "txt")
            
        Returns:
            String representation if output_path is None, else None
        """
        params = self.get_parameter_space()
        
        if format == "txt":
            content = self.describe_parameters()
        elif format == "yaml":
            content = self._to_yaml(params)
        else:  # json
            content = self._to_json(params)
        
        if output_path is not None:
            Path(output_path).write_text(content, encoding="utf-8")
            return None
        return content
    
    def _to_json(self, params: List[ParameterInfo]) -> str:
        """Convert parameters to JSON format."""
        data = {
            "algorithm": self.name,
            "description": f"Tunable hyperparameters for {self.name}",
            "parameters": [p.to_dict() for p in params],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def _to_yaml(self, params: List[ParameterInfo]) -> str:
        """Convert parameters to YAML format (no external dependency)."""
        lines = [
            f"algorithm: {self.name}",
            f"description: Tunable hyperparameters for {self.name}",
            "parameters:",
        ]
        
        for p in params:
            lines.append(f"  - name: {p.name}")
            lines.append(f"    type: {p.type}")
            lines.append(f"    description: \"{p.description}\"")
            
            if p.min_value is not None:
                lines.append(f"    min: {p.min_value}")
            if p.max_value is not None:
                lines.append(f"    max: {p.max_value}")
            if p.choices is not None:
                lines.append(f"    choices: {p.choices}")
            if p.default is not None:
                lines.append(f"    default: {p.default}")
            if p.conditional_on is not None:
                lines.append(f"    conditional_on: {p.conditional_on}")
                lines.append(f"    conditional_value: {p.conditional_value}")
        
        return "\n".join(lines)
    
    def load_reference_front(self, reference_front_file: str) -> np.ndarray:
        """Load reference front from file.
        
        Args:
            reference_front_file: Full filename of the reference front (with extension)
        """
        ref_path = self.reference_fronts_dir / reference_front_file
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference front not found: {ref_path}")
        solutions = read_solutions(str(ref_path))
        return np.array([s.objectives for s in solutions])
    
    def compute_indicators(
        self, 
        front: np.ndarray, 
        reference_front: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute quality indicators for a Pareto front approximation.
        
        Args:
            front: Obtained front as numpy array of objectives
            reference_front: Reference Pareto front
            
        Returns:
            Tuple of (normalized_hypervolume, additive_epsilon)
        """
        nhv_indicator = NormalizedHyperVolume(
            reference_front=reference_front,
            reference_point_offset=self.reference_point_offset,
        )
        nhv_indicator.set_reference_front(reference_front)
        
        epsilon_indicator = AdditiveEpsilonIndicator(reference_front)
        
        nhv_value = float(nhv_indicator.compute(front))
        epsilon_value = float(epsilon_indicator.compute(front))
        
        return nhv_value, epsilon_value
    
    def evaluate(
        self,
        problem: Problem,
        reference_front_file: str,
        params: Dict[str, Any],
        max_evaluations: int,
        n_repeats: int = 1,
    ) -> Tuple[float, float]:
        """
        Evaluate a configuration on a single problem.
        
        Args:
            problem: The optimization problem
            reference_front_file: Full filename of the reference front (with extension)
            params: Hyperparameters to evaluate
            max_evaluations: Maximum evaluations per run
            n_repeats: Number of independent runs
            
        Returns:
            Tuple of mean (normalized_hypervolume, additive_epsilon)
        """
        import copy
        
        reference_front = self.load_reference_front(reference_front_file)
        
        nhv_values = []
        epsilon_values = []
        
        for _ in range(n_repeats):
            # Create and run algorithm
            algorithm = self.create_algorithm(
                copy.deepcopy(problem), params, max_evaluations
            )
            algorithm.run()
            
            # Get non-dominated solutions
            solutions = get_non_dominated_solutions(algorithm.result())
            front = np.array([s.objectives for s in solutions])
            
            # Compute indicators
            nhv, epsilon = self.compute_indicators(front, reference_front)
            nhv_values.append(nhv)
            epsilon_values.append(epsilon)
        
        return float(np.mean(nhv_values)), float(np.mean(epsilon_values))
    
    def evaluate_on_problems(
        self,
        problems: List[Tuple[Problem, str]],
        params: Dict[str, Any],
        max_evaluations: int,
        n_repeats: int = 1,
    ) -> float:
        """
        Evaluate a configuration across multiple problems.
        
        Args:
            problems: List of (problem, reference_front_file) tuples
            params: Hyperparameters to evaluate
            max_evaluations: Maximum evaluations per problem
            n_repeats: Number of independent runs per problem
            
        Returns:
            Aggregated score (lower is better)
        """
        scores = []
        
        for problem, reference_front_file in problems:
            nhv, epsilon = self.evaluate(
                problem, reference_front_file, params, max_evaluations, n_repeats
            )
            # Both indicators: lower is better
            score = nhv + epsilon
            scores.append(score)
        
        return float(np.mean(scores))
    
    def format_params(self, params: Dict[str, Any]) -> str:
        """Format parameters as a human-readable string."""
        parts = []
        for key, value in params.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")
        return ", ".join(parts)
