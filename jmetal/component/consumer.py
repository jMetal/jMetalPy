from jmetal.util.observable import Observer


class BasicSingleObjectiveAlgorithmConsumer(Observer):
    def __init__(self) -> None:
        pass

    def update(self, *args, **kwargs):
        print("Evaluations: " + str(kwargs["evaluations"]) +
              ". Best fitness: " + str(kwargs["best"].objectives[0]) +
              ". Computing time: " + str(kwargs["computing time"]))

