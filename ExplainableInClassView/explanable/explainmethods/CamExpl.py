import skimage

from explanable.explainmethods.AbstractExplainable import AbstractExplainableClass
from explanable.common.utilize import Flatten
import numpy as np
from torch import nn
import torch
from explanable.log.LoggerModule import LoggerModuleClass

log = LoggerModuleClass()
class CamExplClass(AbstractExplainableClass):
    def __init__(self, model: callable or None, device: str = 'cpu', **kwargs):
        super().__init__(model, device, **kwargs)
        self.result = None

    def explain(self, inputs: np.ndarray, **kwargs):

        inputs = torch.tensor(inputs, dtype=torch.float32)
        fourLayers = nn.Sequential(*list(self.model.children())[:-2])(inputs)
        classification = nn.Sequential(*(list(self.model.children())[-2:-1] + [Flatten()]))
        fc = classification(fourLayers)
        _, c, h, w = fourLayers.shape
        temp = (fc @ fourLayers.squeeze(0).reshape(c, h * w)).reshape(h, w)
        self.result = skimage.transform.resize(temp.detach().numpy(), [inputs.shape[-2], inputs.shape[-1]])
        log(f"result: {self.result.shape} inputs: {inputs.shape}")

