import abc

ABC = abc.ABC


class AbstractMetricClass(ABC):
    def __init__(self, model, device: str = 'cpu', **kwargs):
        self.model = model
        self.device = device
        self.explanation = None
        self.predictor = None

    def setExplanation(self, explanation):
        self.explanation = explanation

    def setPredictor(self, predictor):
        self.predictor = predictor

    def evaluate(self, **kwargs):
        raise NotImplementedError