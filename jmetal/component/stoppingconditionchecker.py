class StoppingConditionChecker():
    def __init__(self, **kwargs):
        self.algorithm_data = kwargs

    def check(self) -> bool:
        pass


class StopByEvaluations():
    def __init__(self, **kwargs):
        super(StopAsyncIteration, self).__init(kwargs)

    def check(self) -> bool:
        if not self.algorithm_data["evaluations"]:
            raise Exception("The 'evaluations' field is missing")
        elif not self.algorithm_data["max_evaluations"]:
            raise Exception("The 'max_evaluations' field is missing")

        return self.algorithm_data["evluations"] >= self.algorithm_data["max_evaluations"]