import numpy as np
from jmetal.core.solution import FloatSolution
from jmetal.operator.mutation import UniformMutation
from jmetal.operator.repair import ensure_float_repair


def make_solution(n, lower=0.0, upper=1.0, values=None):
    lbs = [lower] * n
    ubs = [upper] * n
    s = FloatSolution(lbs, ubs, 1)
    if values is None:
        s._variables = [(lower + upper) / 2.0] * n
    else:
        s._variables = list(values)
    return s


def test_uniform_mutation_vectorized_matches_reference():
    n = 20
    prob = 0.5
    perturb = 0.3

    # Prepare deterministic RNG
    seed = 12345
    rng = np.random.default_rng(seed)

    # Create repair operator instance to use both in reference and operator
    repair = ensure_float_repair(None)

    # Initial solution
    vars_init = np.linspace(0.1, 0.9, n)
    solution_vector = make_solution(n, 0.0, 1.0, values=vars_init)
    solution_ref = make_solution(n, 0.0, 1.0, values=vars_init)

    # Create operator with the same RNG
    op = UniformMutation(probability=prob, perturbation=perturb, repair_operator=None, rng=rng)

    # Compute reference result using the same random draws as op (via a new RNG with same seed)
    rng_ref = np.random.default_rng(seed)
    mask = rng_ref.random(n) < prob
    ranges = np.asarray(solution_ref.upper_bound) - np.asarray(solution_ref.lower_bound)
    deltas = (rng_ref.random(n) - 0.5) * perturb * ranges
    candidate = np.where(mask, np.asarray(solution_ref._variables) + deltas, np.asarray(solution_ref._variables))

    # Apply scalar repair per element for reference
    repaired_ref = []
    for val, lb, ub in zip(candidate, solution_ref.lower_bound, solution_ref.upper_bound):
        repaired_ref.append(repair.repair_scalar(float(val), float(lb), float(ub)))

    # Execute vectorized operator (this will use its RNG and repair_vector internally)
    op.execute(solution_vector)

    # Compare
    assert np.allclose(np.asarray(solution_vector._variables), np.asarray(repaired_ref))
