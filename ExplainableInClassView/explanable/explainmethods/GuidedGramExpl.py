from explanable.explainmethods.AbstractExplainable import AbstractExplainableClass
import torch
import numpy as np
import torch.nn as nn


class GuidedGramExplClass(AbstractExplainableClass):
    def __init__(self, model: callable or None, device: str = 'cpu', **kwargs):
        super().__init__(model, device, **kwargs)
        self.result = None

    def explain(self, inputs: np.ndarray, **kwargs) -> None:
        inputTensor = torch.tensor(inputs, requires_grad=True)

        def relu_hook_function(module, grad_in, grad_out):
            if isinstance(module, torch.nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.),)  # min or max, if is in interval, then original value/ RELU

        for i, module in enumerate(self.model.modules()):
            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(relu_hook_function)

        logits = self.model(inputTensor)  # get logits, [bs, num_c]
        index = torch.argmax(logits, axis=1)
        loss = nn.CrossEntropyLoss()
        loss(logits, index).backward()
        gradient = inputTensor.grad
        self.result = self.flatImage(gradient)
