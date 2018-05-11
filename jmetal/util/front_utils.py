from os.path import dirname

from jmetal.core.solution import FloatSolution


def read_front_from_file(file_name: str):
    """ Reads a front from a file and returns a list
    """
    front = []
    with open(file_name) as file:
        for line in file:
            vector = [float(x) for x in line.split()]
            front.append(vector)
    return front


def read_front_from_file_as_solutions(file_name: str):
    """ Reads a front from a file and returns a list of solution objects
    """
    front = []
    with open(file_name) as file:
        for line in file:
            vector = [float(x) for x in line.split()]
            solution = FloatSolution(2, 2, 0, [], [])
            solution.objectives = vector

            front.append(solution)

    return front


def walk_up_folder(path, depth: int =1):
    _cur_depth = 1
    while _cur_depth < depth:
        path = dirname(path)
        _cur_depth += 1
    return path