
from explanable.explainmethods.AbstractExplainable import AbstractExplainableClass
import numpy as np
from torch import nn
import torch
from explanable.log.LoggerModule import LoggerModuleClass
from explanable.common.predictors import predictClassWithIndex
from explanable.common.predictors import predictClassByDefineIndex

log = LoggerModuleClass()

class RiseExplClass(AbstractExplainableClass):
    def __init__(self, model: callable or None, device: str = 'cpu', **kwargs):
        super().__init__(model, device, **kwargs)
        self.result = None
        self.intermediaResults = None

    def explain(self, inputs: np.ndarray, **kwargs):

        assert isinstance(inputs, np.ndarray), log(f"inputs type is: {type(inputs)}")
        assert len(inputs.shape) == 4, log(f"Dimension of inputs: {inputs.shape}")

        generatedSamples = self.generateBlankArea(inputs)
        self.intermediaResults = generatedSamples

        probaByIndex = np.array([])
        probaWithoutIndex = np.array([])
        weighted = [] # dont calculate in numpy type?
        _, index = predictClassWithIndex(self.model, inputs)
        for i in range(generatedSamples.shape[0]):

            prob, _ = predictClassWithIndex(self.model, np.expand_dims(generatedSamples[i], 0))
            probaWithoutIndex = np.append(probaWithoutIndex, prob)

            probaByIndex = np.append(probaByIndex,
                        predictClassByDefineIndex(self.model, np.expand_dims(generatedSamples[i], 0), index))

            prod = probaWithoutIndex[i] * generatedSamples[i]
            weighted.append(prod)

        weightedSum = np.sum(np.array(weighted), axis=0)
        log(f"probaByIndex: {probaByIndex.shape} probaWithoutIndex: {probaWithoutIndex.shape}")
        meanOfRand = np.mean(generatedSamples, axis=0) / generatedSamples.shape[0]
        fs = meanOfRand * weightedSum
        self.result = np.sum(fs, axis=0)
        log(f"result: {self.result.shape} inputs: {inputs.shape}")

    def generateBlankArea(self, inputs: np.ndarray, numOfRand = 20, numOfIntersection = 7, kernelSize: int = None) -> np.ndarray:
        if kernelSize is None:
            kernelSize = 48

        _, _, height, width = inputs.shape

        generatedSamples = []

        numOfRand = numOfRand                   # Number of random sample
        numOfIntersection = numOfIntersection   # Number of rectangle on image

        for _ in range(numOfRand):
            tempData = np.copy(inputs)
            for _ in range(numOfIntersection):
                start_h = int(np.random.randint(0, 224, 1))
                start_w = int(np.random.randint(0, 224, 1))

                end_h = start_h + kernelSize
                if end_h > height:
                    end_h = height

                end_w = start_w + kernelSize
                if end_w > width:
                    end_w = width

                tempData[:, :, start_h:end_h, start_w:end_w] = 127 / 255

            generatedSamples.append(tempData)
        generatedSamples = np.concatenate(generatedSamples)
        return generatedSamples