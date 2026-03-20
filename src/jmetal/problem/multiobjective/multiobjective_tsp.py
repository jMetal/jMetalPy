import math
import re
import os
from pathlib import Path
from typing import List, Tuple

from jmetal.core.problem import PermutationProblem
from jmetal.core.solution import PermutationSolution


class MultiObjectiveTSP(PermutationProblem):
    """Multi-objective TSP problem.

    Reads one or more TSPLIB-like files (NODE_COORD_SECTION) and creates one
    distance matrix per file. All matrices must have the same dimension.

    Usage notes:
    - Passing a single filename produces a single-objective problem (i.e.,
      `number_of_objectives()` will be 1). This makes `MultiObjectiveTSP` a
      drop-in replacement for single-objective TSP instances in most codepaths.
    - Filenames may be given as absolute/relative paths or as short names
      (e.g. "eil101.tsp"); short names are resolved by searching
      `resources/TSP_instances` inside the repository.
    - The reader supports typical TSPLIB `NODE_COORD_SECTION` formats and stops
      at `EOF`/`TOUR_SECTION` markers.
    """

    def __init__(self, distance_files: List[str]):
        super(MultiObjectiveTSP, self).__init__()

        if not distance_files:
            raise ValueError("distance_files must be a non-empty list of file paths")

        self.distance_matrices: List[List[List[float]]] = []
        self.number_of_cities = None

        for f in distance_files:
            resolved = self._resolve_file_path(f)
            matrix, dim = self._read_problem(resolved)
            if self.number_of_cities is None:
                self.number_of_cities = dim
            elif self.number_of_cities != dim:
                raise ValueError("All distance files must have the same DIMENSION")
            self.distance_matrices.append(matrix)

        self.obj_directions = [self.MINIMIZE] * len(self.distance_matrices)

    def number_of_variables(self) -> int:
        return self.number_of_cities

    def number_of_objectives(self) -> int:
        return len(self.distance_matrices)

    def number_of_constraints(self) -> int:
        return 0

    def _read_problem(self, filename: str) -> Tuple[List[List[float]], int]:
        """Read a single TSPLIB-like file and return (matrix, dimension).

        The reader expects a line containing "DIMENSION" and a section with
        lines starting with the city index followed by X and Y coordinates.
        Coordinates are parsed as floats and distances are computed as
        Euclidean distances rounded using int(dist + 0.5) to match the Java
        implementation.
        """
        if filename is None:
            raise FileNotFoundError("Filename can not be None")

        with open(filename) as fh:
            raw_lines = [ln.rstrip("\n") for ln in fh.readlines()]

        # normalize lines (strip but keep empty lines for section detection)
        lines = [ln.strip() for ln in raw_lines if ln.strip()]

        # find dimension
        dim = None
        for ln in lines:
            if ln.upper().startswith("DIMENSION"):
                m = re.search(r"(\d+)", ln)
                if m:
                    dim = int(m.group(1))
                    break

        if dim is None:
            raise ValueError(f"DIMENSION not found in file {filename}")

        coords = [None] * dim

        # Locate the start of coordinates. Prefer explicit NODE_COORD_SECTION
        start_idx = None
        for i, ln in enumerate(lines):
            if ln.upper().startswith("NODE_COORD_SECTION") or ln.upper().startswith("NODE_COORDS_SECTION"):
                start_idx = i + 1
                break

        # If no explicit section marker, fall back to first line that looks like coordinates
        if start_idx is None:
            for i, ln in enumerate(lines):
                if re.match(r"^\d+\s+[-+]?\d", ln):
                    start_idx = i
                    break

        if start_idx is None:
            raise ValueError(f"No coordinate section found in file {filename}")

        # Read coordinates until EOF or until a non-coordinate token
        for ln in lines[start_idx:]:
            up = ln.upper()
            if up.startswith("EOF") or up.startswith("TOUR_SECTION") or up.endswith("SECTION"):
                break
            if re.match(r"^\d+\s+", ln):
                parts = ln.split()
                if len(parts) >= 3:
                    try:
                        idx = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                    except ValueError:
                        continue

                    if 1 <= idx <= dim:
                        coords[idx - 1] = (x, y)

        if any(c is None for c in coords):
            raise ValueError(f"Not all coordinates found in file {filename}")

        # build matrix
        matrix = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            xi, yi = coords[i]
            for j in range(i + 1, dim):
                xj, yj = coords[j]
                dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                dist = int(dist + 0.5)
                matrix[i][j] = dist
                matrix[j][i] = dist

        return matrix, dim

    def _resolve_file_path(self, filename: str) -> str:
        """Resolve filename: if it exists return it, otherwise try resources/TSP_instances."""
        # absolute or existing path
        if os.path.isabs(filename) and os.path.exists(filename):
            return filename

        if os.path.exists(filename):
            return filename

        # try to find in repository resources/TSP_instances
        current = Path(__file__).resolve()
        # parents: multiobjective(0), problem(1), jmetal(2), src(3), repo root(4)
        repo_root = current.parents[4]
        candidate = repo_root / "resources" / "TSP_instances" / filename
        if candidate.exists():
            return str(candidate)

        # try with .tsp extension
        candidate2 = candidate.with_suffix('.tsp')
        if candidate2.exists():
            return str(candidate2)

        raise FileNotFoundError(f"File {filename} not found and not present in resources/TSP_instances")

    def evaluate(self, solution: PermutationSolution) -> PermutationSolution:
        fitness = [0.0] * self.number_of_objectives()

        for i in range(self.number_of_variables() - 1):
            x = solution.variables[i]
            y = solution.variables[i + 1]
            for obj_idx, matrix in enumerate(self.distance_matrices):
                fitness[obj_idx] += matrix[x][y]

        first_city = solution.variables[0]
        last_city = solution.variables[-1]
        for obj_idx, matrix in enumerate(self.distance_matrices):
            fitness[obj_idx] += matrix[first_city][last_city]

        for i in range(self.number_of_objectives()):
            solution.objectives[i] = fitness[i]

        return solution

    def create_solution(self) -> PermutationSolution:
        new_solution = PermutationSolution(
            number_of_variables=self.number_of_variables(),
            number_of_objectives=self.number_of_objectives(),
            number_of_constraints=self.number_of_constraints(),
        )
        # random permutation
        import random

        new_solution.variables = random.sample(range(self.number_of_variables()), k=self.number_of_variables())

        return new_solution

    def name(self):
        return "MultiObjectiveTSP"
