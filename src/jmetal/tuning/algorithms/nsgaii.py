"""
NSGA-II algorithm tuner.

This module provides hyperparameter tuning support for the NSGA-II algorithm.
"""

from typing import Any, Dict, List

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.problem import Problem
from jmetal.operator.crossover import SBXCrossover, BLXAlphaCrossover
from jmetal.operator.mutation import PolynomialMutation, UniformMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

from .base import AlgorithmTuner, ParameterInfo


class NSGAIITuner(AlgorithmTuner):
    """
    Tuner for the NSGA-II algorithm.
    
    Tunable hyperparameters:
        - offspring_population_size: Size of offspring population [1, 10, 50, 100, 150, 200]
        - crossover_type: "sbx" or "blxalpha"
        - crossover_probability: [0.7, 1.0]
        - crossover_eta: SBX distribution index [5, 400]
        - blx_alpha: BLX-alpha parameter [0, 1]
        - mutation_type: "polynomial" or "uniform"
        - mutation_probability_factor: Scales with 1/n_variables [0.5, 2.0]
        - mutation_eta: Polynomial distribution index [5, 400]
        - mutation_perturbation: Uniform mutation perturbation [0.1, 2.0]
    
    Example:
        tuner = NSGAIITuner(population_size=100)
        # Use with tune() function or directly
    """
    
    @property
    def name(self) -> str:
        return "NSGAII"
    
    def get_parameter_space(self) -> List[ParameterInfo]:
        """
        Return the description of all tunable NSGA-II parameters.
        
        Returns:
            List of ParameterInfo describing each hyperparameter
        """
        return [
            # Offspring population size
            ParameterInfo(
                name="offspring_population_size",
                type="categorical",
                description=(
                    "Number of offspring generated in each iteration. "
                    "Higher values explore more solutions per generation but "
                    "increase computational cost. Common values: 100 (balanced), "
                    "200 (more exploration)."
                ),
                choices=[1, 10, 50, 100, 150, 200],
                default=100,
            ),
            
            # Crossover type
            ParameterInfo(
                name="crossover_type",
                type="categorical",
                description=(
                    "Type of crossover operator for combining parent solutions. "
                    "'sbx' (Simulated Binary Crossover): mimics single-point crossover "
                    "for real numbers, good for preserving parent characteristics. "
                    "'blxalpha' (BLX-alpha): creates offspring in an extended range "
                    "around parents, promotes diversity."
                ),
                choices=["sbx", "blxalpha"],
                default="sbx",
            ),
            
            # Crossover probability
            ParameterInfo(
                name="crossover_probability",
                type="float",
                description=(
                    "Probability of applying crossover to a pair of parents. "
                    "Higher values (closer to 1.0) mean more offspring inherit "
                    "traits from both parents. Values of 0.9-1.0 are typical."
                ),
                min_value=0.7,
                max_value=1.0,
                default=0.9,
            ),
            
            # SBX distribution index (conditional)
            ParameterInfo(
                name="crossover_eta",
                type="float",
                description=(
                    "Distribution index for SBX crossover. Controls how similar "
                    "offspring are to parents. Low values (5-20): offspring spread "
                    "further from parents (more exploration). High values (50-400): "
                    "offspring stay closer to parents (more exploitation)."
                ),
                min_value=5.0,
                max_value=400.0,
                default=20.0,
                conditional_on="crossover_type",
                conditional_value="sbx",
            ),
            
            # BLX alpha (conditional)
            ParameterInfo(
                name="blx_alpha",
                type="float",
                description=(
                    "Alpha parameter for BLX-alpha crossover. Controls the extension "
                    "of the range where offspring can be created. 0.0: offspring only "
                    "between parents. 0.5: offspring can extend 50% beyond parents "
                    "(recommended). 1.0: double extension range."
                ),
                min_value=0.0,
                max_value=1.0,
                default=0.5,
                conditional_on="crossover_type",
                conditional_value="blxalpha",
            ),
            
            # Mutation type
            ParameterInfo(
                name="mutation_type",
                type="categorical",
                description=(
                    "Type of mutation operator for introducing variations. "
                    "'polynomial': small perturbations with controlled distribution, "
                    "good for fine-tuning solutions. "
                    "'uniform': random changes within bounds, promotes exploration."
                ),
                choices=["polynomial", "uniform"],
                default="polynomial",
            ),
            
            # Mutation probability factor
            ParameterInfo(
                name="mutation_probability_factor",
                type="float",
                description=(
                    "Factor to compute mutation probability as: factor / n_variables. "
                    "Factor=1.0 means each variable has 1/n chance of mutation "
                    "(standard). Factor=2.0 doubles mutation rate. Factor=0.5 halves it. "
                    "Scales automatically with problem dimensionality."
                ),
                min_value=0.5,
                max_value=2.0,
                default=1.0,
            ),
            
            # Polynomial mutation eta (conditional)
            ParameterInfo(
                name="mutation_eta",
                type="float",
                description=(
                    "Distribution index for polynomial mutation. Controls mutation "
                    "step size. Low values (5-20): larger mutations (more exploration). "
                    "High values (50-400): smaller mutations (fine-tuning). "
                    "Typical value: 20."
                ),
                min_value=5.0,
                max_value=400.0,
                default=20.0,
                conditional_on="mutation_type",
                conditional_value="polynomial",
            ),
            
            # Uniform mutation perturbation (conditional)
            ParameterInfo(
                name="mutation_perturbation",
                type="float",
                description=(
                    "Perturbation factor for uniform mutation. Determines the maximum "
                    "change as a fraction of the variable range. 0.5 means mutations "
                    "can change up to 50% of the variable's range. Higher values "
                    "allow larger jumps in the search space."
                ),
                min_value=0.1,
                max_value=2.0,
                default=0.5,
                conditional_on="mutation_type",
                conditional_value="uniform",
            ),
        ]
    
    def sample_parameters(self, trial, mode: str = "categorical") -> Dict[str, Any]:
        """
        Sample NSGA-II hyperparameters.
        
        Args:
            trial: Optuna trial
            mode: "categorical" for TPE, "continuous" for CMA-ES
            
        Returns:
            Dictionary with sampled parameters
        """
        if mode == "categorical":
            return self._sample_categorical(trial)
        else:
            return self._sample_continuous(trial)
    
    def _sample_categorical(self, trial) -> Dict[str, Any]:
        """Sample using categorical variables (for TPE sampler)."""
        params = {}
        
        # Offspring population size
        params["offspring_population_size"] = trial.suggest_categorical(
            "offspring_population_size", [1, 10, 50, 100, 150, 200]
        )
        
        # Crossover
        params["crossover_type"] = trial.suggest_categorical(
            "crossover_type", ["sbx", "blxalpha"]
        )
        params["crossover_probability"] = trial.suggest_float(
            "crossover_probability", 0.7, 1.0
        )
        
        if params["crossover_type"] == "sbx":
            params["crossover_eta"] = trial.suggest_float("crossover_eta", 5.0, 400.0)
        else:
            params["blx_alpha"] = trial.suggest_float("blx_alpha", 0.0, 1.0)
        
        # Mutation
        params["mutation_type"] = trial.suggest_categorical(
            "mutation_type", ["polynomial", "uniform"]
        )
        params["mutation_probability_factor"] = trial.suggest_float(
            "mutation_probability_factor", 0.5, 2.0
        )
        
        if params["mutation_type"] == "polynomial":
            params["mutation_eta"] = trial.suggest_float("mutation_eta", 5.0, 400.0)
        else:
            params["mutation_perturbation"] = trial.suggest_float(
                "mutation_perturbation", 0.1, 2.0
            )
        
        return params
    
    def _sample_continuous(self, trial) -> Dict[str, Any]:
        """Sample using continuous variables (for CMA-ES sampler)."""
        params = {}
        
        # Offspring as integer with log scale
        params["offspring_population_size"] = trial.suggest_int(
            "offspring_population_size", 1, 200, log=True
        )
        
        # Crossover type as float threshold
        crossover_idx = trial.suggest_float("crossover_type_idx", 0.0, 1.0)
        params["crossover_type"] = "sbx" if crossover_idx < 0.5 else "blxalpha"
        
        params["crossover_probability"] = trial.suggest_float(
            "crossover_probability", 0.7, 1.0
        )
        params["crossover_eta"] = trial.suggest_float("crossover_eta", 5.0, 400.0)
        params["blx_alpha"] = trial.suggest_float("blx_alpha", 0.0, 1.0)
        
        # Mutation type as float threshold
        mutation_idx = trial.suggest_float("mutation_type_idx", 0.0, 1.0)
        params["mutation_type"] = "polynomial" if mutation_idx < 0.5 else "uniform"
        
        params["mutation_probability_factor"] = trial.suggest_float(
            "mutation_probability_factor", 0.5, 2.0
        )
        params["mutation_eta"] = trial.suggest_float("mutation_eta", 5.0, 400.0)
        params["mutation_perturbation"] = trial.suggest_float(
            "mutation_perturbation", 0.1, 2.0
        )
        
        return params
    
    def create_algorithm(
        self,
        problem: Problem,
        params: Dict[str, Any],
        max_evaluations: int
    ) -> NSGAII:
        """
        Create NSGA-II instance with given parameters.
        
        Args:
            problem: Optimization problem
            params: Hyperparameters from sample_parameters()
            max_evaluations: Maximum function evaluations
            
        Returns:
            Configured NSGAII instance
        """
        # Build crossover operator
        crossover = self._build_crossover(params)
        
        # Build mutation operator (probability scales with problem size)
        mutation = self._build_mutation(params, problem.number_of_variables())
        
        # Create algorithm
        return NSGAII(
            problem=problem,
            population_size=self.population_size,
            offspring_population_size=params["offspring_population_size"],
            mutation=mutation,
            crossover=crossover,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    
    def _build_crossover(self, params: Dict[str, Any]):
        """Build crossover operator from parameters."""
        if params["crossover_type"] == "sbx":
            return SBXCrossover(
                probability=params["crossover_probability"],
                distribution_index=params.get("crossover_eta", 20.0),
            )
        else:
            return BLXAlphaCrossover(
                probability=params["crossover_probability"],
                alpha=params.get("blx_alpha", 0.5),
            )
    
    def _build_mutation(self, params: Dict[str, Any], n_variables: int):
        """
        Build mutation operator from parameters.
        
        The actual mutation probability is: factor * (1 / n_variables)
        """
        # Calculate effective probability
        factor = params.get("mutation_probability_factor", 1.0)
        effective_prob = min(1.0, factor / n_variables)
        
        if params.get("mutation_type", "polynomial") == "polynomial":
            return PolynomialMutation(
                probability=effective_prob,
                distribution_index=params.get("mutation_eta", 20.0),
            )
        else:
            return UniformMutation(
                probability=effective_prob,
                perturbation=params.get("mutation_perturbation", 0.5),
            )
    
    def format_params(self, params: Dict[str, Any]) -> str:
        """Format NSGA-II parameters as readable string."""
        offspring = params["offspring_population_size"]
        
        if params["crossover_type"] == "sbx":
            crossover = f"SBX(p={params['crossover_probability']:.3f}, η={params.get('crossover_eta', 20):.1f})"
        else:
            crossover = f"BLX(p={params['crossover_probability']:.3f}, α={params.get('blx_alpha', 0.5):.2f})"
        
        if params.get("mutation_type", "polynomial") == "polynomial":
            mutation = f"Poly(f={params.get('mutation_probability_factor', 1):.2f}, η={params.get('mutation_eta', 20):.1f})"
        else:
            mutation = f"Unif(f={params.get('mutation_probability_factor', 1):.2f}, pert={params.get('mutation_perturbation', 0.5):.2f})"
        
        return f"offspring={offspring}, {crossover}, {mutation}"
