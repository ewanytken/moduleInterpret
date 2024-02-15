import numpy as np
import torch
import skimage.transform
from explanable.common.predictors import predictClassWithIndex
from explanable.common.predictors import predictClassByDefineIndex
from explanable.explainmethods.AbstractExplainable import AbstractExplainableClass
from explanable.metrics.AbstractMetric import AbstractMetricClass


class MorfLerfClass(AbstractMetricClass):
    def __init__(self, model, device: str = 'cpu', **kwargs):
        super().__init__(model, device, **kwargs)
        self.setExplanation(AbstractExplainableClass)
        self.intermediaResults = None

    def evaluate(self, inputs: np.array) -> list:
        self.explanation = self.explanation.result
        self.intermediaResults = self.generateBlankArea(inputs, self.explanation)  # [21 3 224 224]
        return self.AOPCScore(inputs, self.intermediaResults)

    # For one image
    def generateBlankArea(self, inputs: np.ndarray, explanation: np.ndarray,
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

        if explanation.shape[-1] != inputs.shape[-1] or explanation.shape[-2] != inputs.shape[-2]:
            explanation = skimage.transform.resize(explanation, [inputs.shape[-2], inputs.shape[-1]])

        q = 100. / numberOfGenerateSample

        qs = [q * (i - 1) for i in range(numberOfGenerateSample, 0, -1)]
        percentiles = np.percentile(explanation, qs)  # like quantile
        mx = np.array([127 / 255, 127 / 255, 127 / 255]).reshape(3, 1)  # fractional content after preprocessing

        MoRF_images = [inputs]
        fudgedImage = np.copy(inputs)

        for p in percentiles:
            fudgedImage = np.copy(fudgedImage)
            indices = np.where(explanation > p)
            fudgedImage[:, indices[0], indices[1]] = mx
            MoRF_images.append(fudgedImage)
        results['MoRF_images'] = MoRF_images

        LeRF_images = [inputs]
        fudgedImage = inputs.copy()

        for p in percentiles[::-1]:
            fudgedImage = fudgedImage.copy()
            indices = np.where(explanation < p)
            fudgedImage[:, indices[0], indices[1]] = mx
            LeRF_images.append(fudgedImage)
        results['LeRF_images'] = LeRF_images

        # return N sample with blur realm correspondence with percentile.
        # outcoming size: [21 3 224 224] by key dict Morf or Lerf
        return results

    def AOPCScore(self, inputs: np.ndarray, results: dict) -> list:
        probeMorf = []
        probeLerf = []

        _, index = predictClassWithIndex(self.model, inputs)

        for i in range(len(results['LeRF_images'])):
            # expand_dims leads to size [1 c h w]
            probeMorf\
                .append(predictClassByDefineIndex(self.model, np.expand_dims(results['MoRF_images'][i], 0), index).item())
            probeLerf\
                .append(predictClassByDefineIndex(self.model, np.expand_dims(results['LeRF_images'][i], 0), index).item())
        # outcoming size: [21]
        return probeMorf, probeLerf
