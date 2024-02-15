import numpy as np
import torch
import skimage.transform
from explanable.common.predictors import predictClassWithIndex
from explanable.common.predictors import predictClassByDefineIndex
from explanable.explainmethods.AbstractExplainable import AbstractExplainableClass
from explanable.metrics.AbstractMetric import AbstractMetricClass
from explanable.log.LoggerModule import LoggerModuleClass

log = LoggerModuleClass()
class InfidelityClass(AbstractMetricClass):
    def __init__(self, model, device: str = 'cpu', **kwargs):
        super().__init__(model, device, **kwargs)
        self.setPredictor(predictClassByDefineIndex)
        self.setExplanation(AbstractExplainableClass)
        self.generatedSamples = None

    def evaluate(self, inputs: np.ndarray) -> list:
        assert isinstance(inputs, np.ndarray), 'Inputs isnt numpy type'
        self.explanation = self.explanation.result
        isValue, self.generatedSamples = self.generateBlankArea(inputs)
        return self.doIndefidelity(inputs, self.explanation, isValue, self.generatedSamples)

    def generateBlankArea(self, inputs: np.ndarray, kernelSize: list = None) -> np.ndarray:
        if kernelSize is None:
            kernelSize = [32, 64, 128]
        assert isinstance(inputs, np.ndarray), 'Inputs isnt numpy type'

        stride = 8 # stride for kernel moving

        bs, color_channel, height, width = inputs.shape
        isValue = []
        generatedSamples = []

        # Implementation take from paddle - InterpretDL GitHub
        for k in kernelSize:
            h_range = (height - stride) // stride
            w_range = (width - stride) // stride

            if h_range * stride < height:
                h_range += 1
            if w_range * stride < width:
                w_range += 1

            for i in range(h_range):
                start_h = i * stride
                end_h = start_h + k
                if end_h > height:
                    end_h = height
                    break
                for j in range(w_range):
                    start_w = j * stride
                    end_w = start_w + k
                    if end_w > width:
                        end_w = width
                        break
                    tmp_data = np.copy(inputs)
                    tmp_data[:, :, start_h:end_h, start_w:end_w] = 127/255
                    isValue.append(inputs != tmp_data)  # 127 realm on black rectangle [224 244]
                    generatedSamples.append(tmp_data)

        return isValue, generatedSamples

    def doIndefidelity(self, inputs: np.ndarray, explanation: np.ndarray,
                             isValue: list,      generatedSamples: list) -> None:

        assert isinstance(inputs, np.ndarray), 'Inputs isnt numpy type'

        # if len(inputs.shape) == 4:
        #     inputs = inputs.squeeze(0)
        if len(explanation.shape) == 4:
            explanation = np.abs(explanation).squeeze(0).sum(0)
        elif len(explanation.shape) == 3:
            explanation = np.abs(explanation).sum(0)

        if explanation.shape[-1] != inputs.shape[-1]:
            explanation = skimage.transform.resize(explanation, [inputs.shape[-2], inputs.shape[-1]])

        probaInputs, label = predictClassWithIndex(self.model, inputs)
        probaGeneratedSamples = [self.predictor(self.model, generatedSamples[i], label)
                                                    for i in range(len(generatedSamples))]
        probaDiff = probaInputs - probaGeneratedSamples
        probaDiff = np.array(probaDiff)

        squeezeExpl = explanation.reshape(1, 1, explanation.shape[-2], explanation.shape[-1])

        # Calculation takes from article and paddle implementation - GitHub InterpretDL
        explAfterSum = np.sum(isValue * squeezeExpl, axis=(1, 2, 3)) # for default setting [1235]

        # performs optimal scaling for each explanation before calculating the infidelity score
        try:
            beta = (probaDiff * explAfterSum).mean() / np.mean(explAfterSum * explAfterSum)
            explAfterSum *= beta
        except Exception as exception:
            log(exception)

        return np.mean(np.square(probaDiff - explAfterSum))


