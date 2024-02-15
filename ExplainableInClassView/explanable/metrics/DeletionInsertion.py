import numpy as np
import torch
import skimage.transform
from explanable.common.predictors import predictClassWithIndex
from explanable.common.predictors import predictClassByDefineIndex
import explanable.explainmethods.AbstractExplainable
from explanable.metrics.AbstractMetric import AbstractMetricClass

class DeletionInsertionClass(AbstractMetricClass):
    def __init__(self, model, device: str = 'cpu', **kwargs):
        super().__init__(model, device, **kwargs)
        self.setPredictor(predictClassByDefineIndex)
        self.setExplanation(explanable.explainmethods.AbstractExplainable.AbstractExplainableClass)
        self.intermediaResults = None

    def evaluate(self, inputs: np.array) -> list:
        self.explanation = self.explanation.result
        self.intermediaResults = self.generateBlackArea(inputs, self.explanation)  # [21 3 224 224]
        return self.AOPCScore(inputs, self.intermediaResults)

    # For one image
    def generateBlackArea(self, inputs: np.ndarray, explanation: np.ndarray,
                          numberOfGenerateSample: int = 20) -> dict:
        results = {}
        # incoming explanation size: [n_sample, n_channel, h, w]
        # incoming image size:       [n_sample, n_channel, h, w]
        if len(inputs.shape) == 4:
            inputs = inputs.squeeze(0)
        if len(explanation.shape) == 4:
            explanation = np.abs(explanation).squeeze(0).sum(0)
        elif len(explanation.shape) == 3:
            explanation = np.abs(explanation).sum(0)

        if explanation.shape[-1] != inputs.shape[-1]:
            explanation = skimage.transform.resize(explanation, [inputs.shape[-2], inputs.shape[-1]])

        q = 100. / numberOfGenerateSample

        qs = [q * (i - 1) for i in range(numberOfGenerateSample, 0, -1)]
        percentiles = np.percentile(explanation, qs)  # like quantile
        mx = np.array([127 / 255, 127 / 255, 127 / 255]).reshape(3, 1)  # fractional content after preprocessing

        deletionImages = [inputs]
        fudgedImage = np.copy(inputs)

        for p in percentiles:
            fudgedImage = np.copy(fudgedImage)
            indices = np.where(explanation > p)
            fudgedImage[:, indices[0], indices[1]] = mx
            deletionImages.append(fudgedImage)
        results['deletion_images'] = deletionImages

        insertionImages = []
        fudgedImage = np.zeros_like(inputs) + 127/255

        for p in percentiles:
            fudgedImage = fudgedImage.copy()
            indices = np.where(explanation > p)
            fudgedImage[:, indices[0], indices[1]] = inputs[:, indices[0], indices[1]]
            insertionImages.append(fudgedImage)
        insertionImages.append(inputs)
        results['insertion_images'] = insertionImages

        # return N sample with blur realm correspondence with percentile.
        # outcoming size: [21 3 224 224] by key dict Morf or Lerf
        return results

    def AOPCScore(self, inputs: np.ndarray, results: dict) -> list:
        probeDeletion = []
        probeInsertion = []

        _, index = predictClassWithIndex(self.model, inputs)
        for i in range(len(results['deletion_images'])):
            # expand_dims leads to size [1 c h w]
            probeDeletion\
                .append(self.predictor(self.model, np.expand_dims(results['deletion_images'][i], 0), index).item())
            probeInsertion\
                .append(self.predictor(self.model, np.expand_dims(results['insertion_images'][i], 0), index).item())
        # outcoming size: [21]
        return probeDeletion, probeInsertion
