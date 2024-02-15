import numpy as np
import torch
from torch.nn import functional as F
from explanable.explainmethods.AbstractExplainable import AbstractExplainableClass
from tqdm import tqdm

class SmoothGradExplClass(AbstractExplainableClass):
    def __init__(self, model, device: str = 'cpu', **kwargs):
        super().__init__(model, device, **kwargs)
        self.result = None

    def explain(self, numberOfSamples: int, amountNoise: float, inputs: np.ndarray, **kwargs) -> None:
        max_axis = tuple(np.arange(1, inputs.ndim))  # inputs.ndim: 4 (1,2,3)

        stds = amountNoise * (np.max(inputs, axis=max_axis) - np.min(inputs, axis=max_axis))
        totalGradients = np.zeros_like(inputs)
        for i in tqdm(range(numberOfSamples), leave=True, position=0):
            noise = np.concatenate([np.float32(np.random.normal(0.0, stds[j], (1,) + tuple(d.shape))) \
                                    for j, d in enumerate(inputs)])
            _noised_data = inputs + noise
            gradients = self.predictGrad(_noised_data)
            totalGradients += gradients

        self.result = totalGradients / numberOfSamples


