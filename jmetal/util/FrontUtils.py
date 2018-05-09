

def read_front_from_file(file_name: str):
    """ Reads a front from a file and returns a list
    """
    front = []
    file = open(file_name)
    for line in file:
        vector = [float(x) for x in line.split()]
        front.append(vector)

    return front

