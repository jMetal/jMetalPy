class solution:
    _attributes = {}
    def __init__(self, number_of_objectives):
        self.objective = []
        self.number_of_objectives = number_of_objectives

    def set_objective_value(self, index, value):
        self.objective[index] = value

    def get_objective_value(self, index):
        return self.objective[index]

    def get_number_of_objectives(self):
        return len(self.number_of_objectives)

    def set_attribute(self, key, value):
        self._attributes[key] = value

    def get_attribute(self, key):
        return self._attributes[key]