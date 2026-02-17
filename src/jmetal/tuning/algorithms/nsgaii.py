"""
NSGA-II algorithm tuner.

This module provides hyperparameter tuning support for the NSGA-II algorithm.
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.problem import Problem
from jmetal.operator.crossover import SBXCrossover, BLXAlphaCrossover
from jmetal.operator.mutation import PolynomialMutation, UniformMutation
from jmetal.operator.selection import RandomSelection, TournamentSelection
from jmetal.util.archive import CrowdingDistanceArchive, DistanceBasedArchive
from jmetal.util.distance import DistanceMetric
from jmetal.util.evaluator import SequentialEvaluatorWithArchive
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

from .base import AlgorithmTuner, ParameterInfo

if TYPE_CHECKING:
    from jmetal.tuning.tuning_config import ParameterSpaceConfig, ParameterRange


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
        - selection_type: "random" or "tournament"
        - tournament_size: Tournament size k [2, 10] (only if selection_type="tournament")
        - algorithm_result: "population" or "external_archive"
        - archive_type: "crowding_distance" or "distance_based" (only if algorithm_result="external_archive")
        - population_size_with_archive: [10, 200] (only if algorithm_result="external_archive")
    
    The search space can be customized via a ParameterSpaceConfig from YAML.
    
    Example:
        tuner = NSGAIITuner(population_size=100)
        # Use with tune() function or directly
        
        # With custom parameter space from YAML config
        tuner = NSGAIITuner(population_size=100, parameter_space=config.parameter_space)
    """
    
    def __init__(
        self,
        population_size: int = 100,
        parameter_space: Optional["ParameterSpaceConfig"] = None,
    ):
        """
        Initialize NSGA-II tuner.
        
        Args:
            population_size: Fixed population size for NSGA-II
            parameter_space: Optional custom parameter space configuration from YAML
        """
        super().__init__(population_size=population_size)
        self._parameter_space = parameter_space
    
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
            
            # Selection type
            ParameterInfo(
                name="selection_type",
                type="categorical",
                description=(
                    "Type of selection operator for choosing parents. "
                    "'random': uniform random selection, no selection pressure. "
                    "'tournament': k-ary tournament selection, applies selection "
                    "pressure favoring better solutions."
                ),
                choices=["random", "tournament"],
                default="tournament",
            ),
            
            # Tournament size (conditional)
            ParameterInfo(
                name="tournament_size",
                type="int",
                description=(
                    "Number of solutions competing in each tournament (k). "
                    "Higher k increases selection pressure: k=2 is mild pressure, "
                    "k=10 strongly favors best solutions. Only used when "
                    "selection_type='tournament'."
                ),
                min_value=2,
                max_value=10,
                default=2,
                conditional_on="selection_type",
                conditional_value="tournament",
            ),
            
            # Algorithm result type
            ParameterInfo(
                name="algorithm_result",
                type="categorical",
                description=(
                    "Where to get the final result from. "
                    "'population': use the final population (standard NSGA-II). "
                    "'external_archive': use an external archive that stores "
                    "non-dominated solutions found during the entire run."
                ),
                choices=["population", "external_archive"],
                default="population",
            ),
            
            # Archive type (conditional on algorithm_result)
            ParameterInfo(
                name="archive_type",
                type="categorical",
                description=(
                    "Type of external archive. "
                    "'crowding_distance': uses crowding distance for diversity. "
                    "'distance_based': uses L2 squared distance metric."
                ),
                choices=["crowding_distance", "distance_based"],
                default="crowding_distance",
                conditional_on="algorithm_result",
                conditional_value="external_archive",
            ),
            
            # Population size with archive (conditional)
            ParameterInfo(
                name="population_size_with_archive",
                type="int",
                description=(
                    "Population size when using external archive. The archive size "
                    "is fixed to the tuner's population_size, but the algorithm's "
                    "population can be tuned independently for efficiency."
                ),
                min_value=10,
                max_value=200,
                default=100,
                conditional_on="algorithm_result",
                conditional_value="external_archive",
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
    
    def _get_range(self, value) -> tuple:
        """Extract min/max from ParameterRange or return fixed value as range."""
        from jmetal.tuning.tuning_config import ParameterRange
        if isinstance(value, ParameterRange):
            return value.min, value.max
        else:
            # Fixed value - return as both min and max
            return float(value), float(value)
    
    def _get_int_range(self, value) -> tuple:
        """Extract min/max integers from ParameterRange or return fixed value as range."""
        from jmetal.tuning.tuning_config import ParameterRange
        if isinstance(value, ParameterRange):
            return int(value.min), int(value.max)
        else:
            # Fixed value - return as both min and max
            return int(value), int(value)
    
    def _sample_categorical(self, trial) -> Dict[str, Any]:
        """Sample using categorical variables (for TPE sampler)."""
        params = {}
        ps = self._parameter_space  # Shorthand
        
        # Offspring population size
        if ps is not None:
            from jmetal.tuning.tuning_config import CategoricalParameter
            ops = ps.offspring_population_size
            if isinstance(ops, CategoricalParameter):
                offspring_values = ops.values
            elif isinstance(ops, list):
                offspring_values = ops
            else:
                offspring_values = [ops]  # Single fixed value
        else:
            offspring_values = [1, 10, 50, 100, 150, 200]
        
        params["offspring_population_size"] = trial.suggest_categorical(
            "offspring_population_size", offspring_values
        )
        
        # Crossover type
        if ps is not None:
            crossover_types = ps.crossover.types
        else:
            crossover_types = ["sbx", "blxalpha"]
        
        if len(crossover_types) == 1:
            params["crossover_type"] = crossover_types[0]
        else:
            params["crossover_type"] = trial.suggest_categorical(
                "crossover_type", crossover_types
            )
        
        # Crossover probability
        if ps is not None:
            prob_min, prob_max = self._get_range(ps.crossover.probability)
        else:
            prob_min, prob_max = 0.7, 1.0
        
        if prob_min == prob_max:
            params["crossover_probability"] = prob_min
        else:
            params["crossover_probability"] = trial.suggest_float(
                "crossover_probability", prob_min, prob_max
            )
        
        # Crossover-specific parameters
        if params["crossover_type"] == "sbx":
            if ps is not None:
                eta_min, eta_max = self._get_range(ps.crossover.sbx_distribution_index)
            else:
                eta_min, eta_max = 5.0, 400.0
            
            if eta_min == eta_max:
                params["crossover_eta"] = eta_min
            else:
                params["crossover_eta"] = trial.suggest_float(
                    "crossover_eta", eta_min, eta_max
                )
        else:  # blxalpha
            if ps is not None:
                alpha_min, alpha_max = self._get_range(ps.crossover.blx_alpha)
            else:
                alpha_min, alpha_max = 0.0, 1.0
            
            if alpha_min == alpha_max:
                params["blx_alpha"] = alpha_min
            else:
                params["blx_alpha"] = trial.suggest_float(
                    "blx_alpha", alpha_min, alpha_max
                )
        
        # Mutation type
        if ps is not None:
            mutation_types = ps.mutation.types
        else:
            mutation_types = ["polynomial", "uniform"]
        
        if len(mutation_types) == 1:
            params["mutation_type"] = mutation_types[0]
        else:
            params["mutation_type"] = trial.suggest_categorical(
                "mutation_type", mutation_types
            )
        
        # Mutation probability factor
        if ps is not None:
            factor_min, factor_max = self._get_range(ps.mutation.probability_factor)
        else:
            factor_min, factor_max = 0.5, 2.0
        
        if factor_min == factor_max:
            params["mutation_probability_factor"] = factor_min
        else:
            params["mutation_probability_factor"] = trial.suggest_float(
                "mutation_probability_factor", factor_min, factor_max
            )
        
        # Mutation-specific parameters
        if params["mutation_type"] == "polynomial":
            if ps is not None:
                eta_min, eta_max = self._get_range(ps.mutation.polynomial_distribution_index)
            else:
                eta_min, eta_max = 5.0, 400.0
            
            if eta_min == eta_max:
                params["mutation_eta"] = eta_min
            else:
                params["mutation_eta"] = trial.suggest_float(
                    "mutation_eta", eta_min, eta_max
                )
        else:  # uniform
            if ps is not None:
                pert_min, pert_max = self._get_range(ps.mutation.uniform_perturbation)
            else:
                pert_min, pert_max = 0.1, 2.0
            
            if pert_min == pert_max:
                params["mutation_perturbation"] = pert_min
            else:
                params["mutation_perturbation"] = trial.suggest_float(
                    "mutation_perturbation", pert_min, pert_max
                )
        
        # Selection type
        if ps is not None:
            selection_types = ps.selection.types
        else:
            selection_types = ["random", "tournament"]
        
        if len(selection_types) == 1:
            params["selection_type"] = selection_types[0]
        else:
            params["selection_type"] = trial.suggest_categorical(
                "selection_type", selection_types
            )
        
        # Tournament-specific parameters (only if tournament selected)
        if params["selection_type"] == "tournament":
            if ps is not None:
                size_min, size_max = self._get_int_range(ps.selection.tournament.size)
            else:
                size_min, size_max = 2, 10
            
            if size_min == size_max:
                params["tournament_size"] = size_min
            else:
                params["tournament_size"] = trial.suggest_int(
                    "tournament_size", size_min, size_max
                )
        
        # Algorithm result type
        if ps is not None:
            result_types = ps.algorithm_result.types
        else:
            result_types = ["population", "external_archive"]
        
        if len(result_types) == 1:
            params["algorithm_result"] = result_types[0]
        else:
            params["algorithm_result"] = trial.suggest_categorical(
                "algorithm_result", result_types
            )
        
        # Archive-specific parameters (only if external_archive selected)
        if params["algorithm_result"] == "external_archive":
            if ps is not None:
                archive_types = ps.algorithm_result.external_archive.archive_types
            else:
                archive_types = ["crowding_distance", "distance_based"]
            
            if len(archive_types) == 1:
                params["archive_type"] = archive_types[0]
            else:
                params["archive_type"] = trial.suggest_categorical(
                    "archive_type", archive_types
                )
            
            # Population size with archive
            if ps is not None:
                pop_min, pop_max = self._get_int_range(
                    ps.algorithm_result.external_archive.population_size_with_archive
                )
            else:
                pop_min, pop_max = 10, 200
            
            if pop_min == pop_max:
                params["population_size_with_archive"] = pop_min
            else:
                params["population_size_with_archive"] = trial.suggest_int(
                    "population_size_with_archive", pop_min, pop_max
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
        
        # Selection type as float threshold
        selection_idx = trial.suggest_float("selection_type_idx", 0.0, 1.0)
        params["selection_type"] = "random" if selection_idx < 0.5 else "tournament"
        params["tournament_size"] = trial.suggest_int("tournament_size", 2, 10)
        
        # Algorithm result type as float threshold
        result_idx = trial.suggest_float("algorithm_result_idx", 0.0, 1.0)
        params["algorithm_result"] = "population" if result_idx < 0.5 else "external_archive"
        
        # Archive parameters (always sampled in continuous mode)
        archive_idx = trial.suggest_float("archive_type_idx", 0.0, 1.0)
        params["archive_type"] = "crowding_distance" if archive_idx < 0.5 else "distance_based"
        params["population_size_with_archive"] = trial.suggest_int(
            "population_size_with_archive", 10, 200
        )
        
        return params
    
    def create_algorithm(
        self,
        problem: Problem,
        params: Dict[str, Any],
        max_evaluations: int
    ) -> Tuple[NSGAII, Optional[SequentialEvaluatorWithArchive]]:
        """
        Create NSGA-II instance with given parameters.
        
        Args:
            problem: Optimization problem
            params: Hyperparameters from sample_parameters()
            max_evaluations: Maximum function evaluations
            
        Returns:
            Tuple of (NSGAII instance, evaluator with archive or None)
            If algorithm_result is "external_archive", returns the evaluator
            so the caller can get results from the archive.
        """
        # Build crossover operator
        crossover = self._build_crossover(params)
        
        # Build mutation operator (probability scales with problem size)
        mutation = self._build_mutation(params, problem.number_of_variables())
        
        # Build selection operator
        selection = self._build_selection(params)
        
        # Determine if using external archive
        algorithm_result = params.get("algorithm_result", "population")
        evaluator = None
        population_size = self.population_size
        
        if algorithm_result == "external_archive":
            # Create archive with size = tuner's population_size
            archive = self._build_archive(params, self.population_size)
            evaluator = SequentialEvaluatorWithArchive(archive)
            # Use the tuned population size for the algorithm
            population_size = params.get("population_size_with_archive", self.population_size)
        
        # Create algorithm - only pass evaluator if using external archive
        if evaluator is not None:
            algorithm = NSGAII(
                problem=problem,
                population_size=population_size,
                offspring_population_size=params["offspring_population_size"],
                mutation=mutation,
                crossover=crossover,
                selection=selection,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
                population_evaluator=evaluator,
            )
        else:
            algorithm = NSGAII(
                problem=problem,
                population_size=population_size,
                offspring_population_size=params["offspring_population_size"],
                mutation=mutation,
                crossover=crossover,
                selection=selection,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            )
        
        return algorithm, evaluator
    
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
    
    def _build_selection(self, params: Dict[str, Any]):
        """Build selection operator from parameters."""
        selection_type = params.get("selection_type", "tournament")
        
        if selection_type == "random":
            return RandomSelection()
        else:  # tournament
            tournament_size = params.get("tournament_size", 2)
            return TournamentSelection(tournament_size=tournament_size)
    
    def _build_archive(self, params: Dict[str, Any], archive_size: int):
        """
        Build external archive from parameters.
        
        Args:
            params: Hyperparameters containing archive_type
            archive_size: Maximum size of the archive
            
        Returns:
            Archive instance (CrowdingDistanceArchive or DistanceBasedArchive)
        """
        archive_type = params.get("archive_type", "crowding_distance")
        
        if archive_type == "crowding_distance":
            return CrowdingDistanceArchive(maximum_size=archive_size)
        else:  # distance_based
            return DistanceBasedArchive(
                maximum_size=archive_size,
                metric=DistanceMetric.L2_SQUARED
            )
    
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
        
        Overrides base class to handle external archive results.
        
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
            algorithm, evaluator = self.create_algorithm(
                copy.deepcopy(problem), params, max_evaluations
            )
            algorithm.run()
            
            # Get results: from archive if using external_archive, else from algorithm
            algorithm_result = params.get("algorithm_result", "population")
            if algorithm_result == "external_archive" and evaluator is not None:
                solutions = get_non_dominated_solutions(evaluator.get_archive().solution_list)
            else:
                solutions = get_non_dominated_solutions(algorithm.result())
            
            front = np.array([s.objectives for s in solutions])
            
            # Compute indicators
            nhv, epsilon = self.compute_indicators(front, reference_front)
            nhv_values.append(nhv)
            epsilon_values.append(epsilon)
        
        return float(np.mean(nhv_values)), float(np.mean(epsilon_values))
    
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
        
        selection_type = params.get("selection_type", "tournament")
        if selection_type == "random":
            selection = "Random"
        else:
            selection = f"Tournament(k={params.get('tournament_size', 2)})"
        
        # Build result string
        parts = [f"offspring={offspring}", crossover, mutation, selection]
        
        # Add algorithm_result info if using external archive
        algorithm_result = params.get("algorithm_result", "population")
        if algorithm_result == "external_archive":
            archive_type = params.get("archive_type", "crowding_distance")
            pop_size = params.get("population_size_with_archive", self.population_size)
            archive_info = f"Archive({archive_type}, pop={pop_size})"
            parts.append(archive_info)
        
        return ", ".join(parts)
