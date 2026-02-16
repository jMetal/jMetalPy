"""Tuning de NSGA-II para ZDT1: Entrenamiento y Validación.

Espacio de parámetros extendido:
- Crossover: SBX (probability, distribution_index)
- Mutation: Polynomial (probability, distribution_index)  
- Offspring population size: [0, 10, 20, 50, 100, 200] (0 = same as population)
- Algorithm result: population | externalArchive (with CrowdingDistanceArchive)
- Population size with archive: [10, 200] (only when using archive)

Configuración:
- Entrenamiento: 10,000 evaluaciones
- Validación: 20,000 evaluaciones  
- Paralelismo: 8 cores
- Problema: ZDT1

Uso:
    python tune_nsgaii_zdt1_test.py
"""

import logging
import time
import warnings
from pathlib import Path

import numpy as np

# Suppress verbose logging
logging.getLogger("optuna").setLevel(logging.WARNING)
logging.getLogger("jmetal").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.problem import ZDT1
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.comparator import DominanceWithConstraintsComparator
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.evaluator import SequentialEvaluatorWithArchive

from concurrent.futures import ProcessPoolExecutor, as_completed

from jmetal.tuning import (
    TuningProtocol,
    TuningConfig,
    ParameterSpace,
)
from jmetal.tuning.indicators import IndicatorSet


# Top-level function for parallel execution (must be pickle-able)
def _run_single_extended_parallel(
    problem,
    config: dict,
    max_evaluations: int,
    population_size: int,
    seed: int,
) -> np.ndarray:
    """Execute a single run with extended parameters (for parallel execution)."""
    from jmetal.algorithm.multiobjective.nsgaii import NSGAII
    from jmetal.operator.crossover import SBXCrossover, BLXAlphaCrossover
    from jmetal.operator.mutation import PolynomialMutation, UniformMutation
    from jmetal.operator.selection import BinaryTournamentSelection, RandomSelection
    from jmetal.util.termination_criterion import StoppingByEvaluations
    from jmetal.util.comparator import DominanceWithConstraintsComparator
    from jmetal.util.solution import get_non_dominated_solutions
    from jmetal.util.archive import CrowdingDistanceArchive
    from jmetal.util.evaluator import SequentialEvaluatorWithArchive
    
    # Helper to get nested config values
    def get_param(config, *keys, default=None):
        """Get nested parameter value."""
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    # Determine if using archive
    algorithm_result = config.get("algorithmResult", {})
    if isinstance(algorithm_result, dict):
        use_archive = algorithm_result.get("choice") == "externalArchive"
        pop_size_with_archive = algorithm_result.get("populationSizeWithArchive", 100)
        archive_type = get_param(algorithm_result, "archiveType", "choice", default="crowdingDistanceArchive")
    else:
        use_archive = False
        pop_size_with_archive = 100
        archive_type = "crowdingDistanceArchive"
    
    # Determine population size
    if use_archive:
        pop_size = pop_size_with_archive
    else:
        pop_size = population_size
    
    # Determine offspring size
    offspring_size = config.get("offspringPopulationSize", 100)
    
    # Build crossover operator
    crossover_config = config.get("crossover", {})
    crossover_type = crossover_config.get("choice", "SBX")
    crossover_prob = crossover_config.get("probability", 0.9)
    
    if crossover_type == "SBX":
        crossover = SBXCrossover(
            probability=crossover_prob,
            distribution_index=crossover_config.get("distributionIndex", 20.0),
        )
    elif crossover_type == "BLXAlpha":
        crossover = BLXAlphaCrossover(
            probability=crossover_prob,
            alpha=crossover_config.get("alpha", 0.5),
        )
    else:
        crossover = SBXCrossover(probability=crossover_prob, distribution_index=20.0)
    
    # Build mutation operator
    mutation_config = config.get("mutation", {})
    mutation_type = mutation_config.get("choice", "Polynomial")
    prob_factor = mutation_config.get("probabilityFactor", 1.0)
    mutation_prob = prob_factor / problem.number_of_variables()
    
    if mutation_type == "Polynomial":
        mutation = PolynomialMutation(
            probability=mutation_prob,
            distribution_index=mutation_config.get("distributionIndex", 20.0),
        )
    elif mutation_type == "Uniform":
        mutation = UniformMutation(
            probability=mutation_prob,
            perturbation=mutation_config.get("perturbation", 0.5),
        )
    else:
        mutation = PolynomialMutation(probability=mutation_prob, distribution_index=20.0)
    
    # Build selection operator
    selection_config = config.get("selection", {})
    selection_type = selection_config.get("choice", "BinaryTournament")
    
    if selection_type == "Random":
        selection = RandomSelection()
    else:
        selection = BinaryTournamentSelection(DominanceWithConstraintsComparator())
    
    # Build algorithm with or without archive
    if use_archive:
        archive = CrowdingDistanceArchive(maximum_size=100)
        evaluator = SequentialEvaluatorWithArchive(archive)
        
        algorithm = NSGAII(
            problem=problem,
            population_size=pop_size,
            offspring_population_size=offspring_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            dominance_comparator=DominanceWithConstraintsComparator(),
            population_evaluator=evaluator,
        )
        
        algorithm.run()
        solutions = archive.solution_list
    else:
        algorithm = NSGAII(
            problem=problem,
            population_size=pop_size,
            offspring_population_size=offspring_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            dominance_comparator=DominanceWithConstraintsComparator(),
        )
        
        algorithm.run()
        solutions = get_non_dominated_solutions(algorithm.result())
    
    return np.array([s.objectives for s in solutions])


class TuningProtocolExtended(TuningProtocol):
    """TuningProtocol extendido que soporta archivo externo y offspring variable."""
    
    def objective(self, trial) -> float:
        """Objective function que maneja parámetros extendidos.
        
        Soporta:
        - offspringPopulationSize
        - algorithmResult: population | externalArchive
        - populationSizeWithArchive (only with archive)
        """
        # Sample configuration
        config = self.parameter_space.sample(trial)
        
        # Use parallel execution if enabled
        if self.config.parallel_runs:
            return self._objective_parallel_extended(trial, config)
        else:
            return self._objective_sequential_extended(trial, config)
    
    def _objective_parallel_extended(self, trial, config: dict) -> float:
        """Parallel execution of runs using ProcessPoolExecutor."""
        # Prepare all (problem, run_idx, seed) combinations
        tasks = []
        for problem in self.problems:
            for run_idx in range(self.config.n_repeats):
                seed = self._compute_seed(trial.number, run_idx)
                tasks.append((problem, run_idx, seed))
        
        # Execute runs in parallel
        results = []
        n_workers = self.config.n_workers or None
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _run_single_extended_parallel,
                    problem,
                    config,
                    self.config.max_evaluations,
                    self.config.population_size,
                    seed,
                ): (problem.name(), run_idx)
                for problem, run_idx, seed in tasks
            }
            
            for future in as_completed(futures):
                problem_name, run_idx = futures[future]
                objectives = future.result()
                results.append((problem_name, objectives))
        
        # Compute indicators and score
        problem_indicator_values = {
            p.name(): {name: [] for name in self.config.indicator_names}
            for p in self.problems
        }
        
        for problem_name, objectives in results:
            indicator_set = self._indicator_sets[problem_name]
            indicators = indicator_set.compute(objectives)
            for name, value in indicators.items():
                problem_indicator_values[problem_name][name].append(value)
        
        # Compute score
        score = 0.0
        for problem_name, indicator_values in problem_indicator_values.items():
            for indicator_name, values in indicator_values.items():
                mean_value = float(np.mean(values))
                score += mean_value
        
        return score
    
    def _objective_sequential_extended(self, trial, config: dict) -> float:
        """Sequential execution of runs."""
        problem_indicator_values = {}
        
        for problem in self.problems:
            problem_name = problem.name()
            indicator_set = self._indicator_sets[problem_name]
            
            problem_indicator_values[problem_name] = {
                name: [] for name in self.config.indicator_names
            }
            
            for run_idx in range(self.config.n_repeats):
                seed = self._compute_seed(trial.number, run_idx)
                
                objectives = self._run_single_extended(
                    problem=problem,
                    config=config,
                    max_evaluations=self.config.max_evaluations,
                    seed=seed,
                )
                
                indicators = indicator_set.compute(objectives)
                for name, value in indicators.items():
                    problem_indicator_values[problem_name][name].append(value)
        
        score = 0.0
        for problem_name, indicator_values in problem_indicator_values.items():
            for indicator_name, values in indicator_values.items():
                mean_value = float(np.mean(values))
                score += mean_value
        
        return score
    
    def _run_single_extended(
        self,
        problem,
        config: dict,
        max_evaluations: int,
        seed: int,
    ) -> np.ndarray:
        """Execute a single run with the given configuration (sequential mode)."""
        # Simply delegate to the parallel function (it works for sequential too)
        return _run_single_extended_parallel(
            problem=problem,
            config=config,
            max_evaluations=max_evaluations,
            population_size=self.config.population_size,
            seed=seed,
        )


def run_validation(
    best_params: dict,
    problem,
    reference_front: np.ndarray,
    max_evaluations: int = 20000,
    n_runs: int = 5,
    base_seed: int = 9999,
    population_size: int = 100,
):
    """Ejecuta validación con la mejor configuración encontrada."""
    print("\n" + "=" * 60)
    print("FASE DE VALIDACIÓN")
    print("=" * 60)
    print(f"Evaluaciones: {max_evaluations}")
    print(f"Runs: {n_runs}")
    
    indicator_set = IndicatorSet(
        indicator_names=["NHV", "Epsilon"],
        reference_front=reference_front,
        reference_point_offset=0.1,
    )
    
    results = {"NHV": [], "Epsilon": []}
    
    for run in range(n_runs):
        seed = base_seed + run
        
        # Use the same function as tuning
        objectives = _run_single_extended_parallel(
            problem=problem,
            config=best_params,
            max_evaluations=max_evaluations,
            population_size=population_size,
            seed=seed,
        )
        
        indicators = indicator_set.compute(objectives)
        for name, value in indicators.items():
            results[name].append(value)
        
        # Determine mode for display
        algorithm_result = best_params.get("algorithmResult", {})
        if isinstance(algorithm_result, dict):
            use_archive = algorithm_result.get("choice") == "externalArchive"
        else:
            use_archive = False
        mode_str = "archive" if use_archive else "population"
        
        print(f"  Run {run + 1} ({mode_str}): NHV={indicators['NHV']:.6f}, Epsilon={indicators['Epsilon']:.6f}")
    
    print("\n" + "-" * 40)
    print("Resultados de validación (media ± std):")
    for name, values in results.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"  {name}: {mean:.6f} ± {std:.6f}")
    
    return results


def main():
    print("=" * 60)
    print("TUNING DE NSGA-II PARA ZDT1")
    print("=" * 60)
    
    # Configuración
    problems = [ZDT1()]
    
    # Cargar frente de referencia
    fronts_dir = Path(__file__).parent.parent.parent.parent / "resources" / "reference_fronts"
    reference_fronts = TuningProtocol.load_reference_fronts(problems, fronts_dir)
    
    # Cargar espacio de parámetros desde archivo YAML
    yaml_path = Path(__file__).parent.parent.parent.parent / "src" / "jmetal" / "tuning" / "parameter_spaces" / "NSGAIIFloat.yaml"
    parameter_space = ParameterSpace.from_yaml(yaml_path)
    
    print(f"\nEspacio de parámetros cargado desde: {yaml_path.name}")
    
    # Configuración de tuning
    # - Entrenamiento: 10,000 evaluaciones
    # - 8 cores para paralelismo
    config = TuningConfig(
        n_repeats=1,              # 1 repetición (pruebas preliminares)
        population_size=100,
        max_evaluations=10000,    # Entrenamiento: 10k evaluaciones
        base_seed=42,
        indicator_names=["NHV", "Epsilon"],  # NHV y Epsilon
        parallel_runs=True,       # Paralelismo habilitado
        n_workers=8,              # 8 cores
    )
    
    print(f"\nConfiguración de entrenamiento:")
    print(f"  - Problema: ZDT1")
    print(f"  - Evaluaciones: {config.max_evaluations}")
    print(f"  - Repeticiones: {config.n_repeats}")
    print(f"  - Workers: {config.n_workers}")
    print(f"  - Indicadores: {list(config.indicator_names)}")
    
    # Crear protocolo con builder personalizado
    protocol = TuningProtocolExtended(
        algorithm_class=NSGAII,
        parameter_space=parameter_space,
        problems=problems,
        reference_fronts=reference_fronts,
        config=config,
        artifact_dir="artifacts_zdt1_tuning_extended",
        study_name="nsgaii_zdt1_extended",
    )
    
    # Crear estudio
    study = protocol.create_study(
        storage="sqlite:///nsgaii_zdt1_tuning_extended.db",
        load_if_exists=False,
    )
    
    # Callback para mostrar progreso
    def trial_callback(study, trial):
        """Muestra el progreso después de cada trial."""
        # Mostrar trial actual
        print(f"\nTrial {trial.number + 1}: score = {trial.value:.6f}")
        
        # Mostrar configuración de este trial
        for key, value in trial.params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Mostrar mejor hasta ahora
        if study.best_trial.number == trial.number:
            print("  ** NUEVO MEJOR **")
    
    # Fase de entrenamiento
    n_trials = 200  # 200 configuraciones
    print(f"\n{'=' * 60}")
    print(f"FASE DE ENTRENAMIENTO: {n_trials} trials")
    print("=" * 60)
    
    start_time = time.perf_counter()
    study.optimize(
        protocol.objective,
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=False,
        callbacks=[trial_callback],
    )
    training_time = time.perf_counter() - start_time
    
    # Resultados del entrenamiento
    print(f"\n{'=' * 60}")
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("=" * 60)
    print(f"Tiempo total: {training_time:.2f} segundos")
    print(f"Tiempo por trial: {training_time / n_trials:.2f} segundos")
    print(f"Mejor score: {study.best_value:.6f}")
    print("\nMejor configuración encontrada:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Guardar resumen
    summary_path = protocol.save_summary(study, training_time)
    if summary_path:
        print(f"\nResumen guardado en: {summary_path}")
    
    # Fase de validación con 20,000 evaluaciones
    validation_results = run_validation(
        best_params=study.best_params,
        problem=ZDT1(),
        reference_front=reference_fronts["ZDT1"],
        max_evaluations=20000,  # Validación: 20k evaluaciones
        n_runs=5,
        base_seed=9999,
    )
    
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"Mejor configuración:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(f"\nValidación (20k evaluaciones, 5 runs):")
    print(f"  NHV: {np.mean(validation_results['NHV']):.6f} ± {np.std(validation_results['NHV']):.6f}")
    print(f"  Epsilon: {np.mean(validation_results['Epsilon']):.6f} ± {np.std(validation_results['Epsilon']):.6f}")


if __name__ == "__main__":
    main()
