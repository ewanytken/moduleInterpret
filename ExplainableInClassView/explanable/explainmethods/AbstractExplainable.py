import abc
import torch
import numpy as np
from torch.nn import functional as F
import logging
ABC = abc.ABC


class AbstractExplainableClass(ABC):
    def __init__(self, model, device: str = 'cpu', **kwargs):
        self.model = model
        self.device = device
        self.predictor = None


    def predictGrad(self, inputs: np.ndarray) -> np.ndarray:
        inputTensor = torch.tensor(inputs, requires_grad=True)
        logits = self.model(inputTensor)
        num_classes = logits.shape[1]  # 1000
        probas = F.softmax(logits, dim=1)  # get probabilities.
        label = torch.argmax(logits, axis=1).numpy()  # for example: 281 - tabby cat
        labels_onehot = torch.tensor(torch.nn.functional.one_hot(torch.tensor(label),
                                                                 num_classes=num_classes),
                                                                 dtype=torch.float64,
                                                                 requires_grad=True)  # for stack from distinct image. Dont need if one image
        loss = torch.sum(probas * labels_onehot)  # for example 281 to 0.384 probability
        gradients = torch.autograd.grad(loss, inputTensor)
        
        return gradients[0].detach().numpy()

    def flatImage(self, gradient):
        gradient = np.array(gradient)
        gradNorm = gradient.squeeze(0).transpose(1,2,0)                    # 224 224 3
        gradNorm = gradNorm[:,:,0]+ gradNorm[:,:,1]+ gradNorm[:,:,2]       # 224 224 sum
        gradNorm = (gradNorm - np.min(gradNorm))/ (np.max(gradNorm)- np.min(gradNorm))
        return gradNorm

    def setPredictor(self, predictor):
        self.predictor = predictor

    def explain(self, **kwargs):
        raise NotImplementedError