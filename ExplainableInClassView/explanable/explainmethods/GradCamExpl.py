from explanable.explainmethods.AbstractExplainable import AbstractExplainableClass
import torch
from torch import nn
import numpy as np
from explanable.common.utilize import Flatten
from explanable.common.predictors import predictClassWithIndex
from explanable.common.utilize import VitTransformerChanger
import skimage
import torch.nn.functional as F
import transformers

class GradCamExplClass(AbstractExplainableClass):
    def __init__(self, model: callable or None, device: str = 'cpu', **kwargs):
        super().__init__(model, device, **kwargs)
        self.result = None
        self.setPredictor(predictClassWithIndex)

    def explain(self, inputs: np.ndarray, **kwargs):
        inputs = torch.tensor(inputs, dtype=torch.float32)
        _, index = self.predictor(self.model, inputs)

        if type(self.model) == transformers.models.vit.modeling_vit.ViTForImageClassification:
            lastLayers = nn.Sequential(*list(self.model.children())[-1:])                    # ONLY LAST LAYER torch.Size([1, 197, 1000])
            beforeLastLayer = nn.Sequential(*list(self.model.children())[:-1] + [VitTransformerChanger()]) # torch.Size([1, 197, 768])
            lx197x768 = beforeLastLayer(inputs)
            lx197x1000 = lastLayers(lx197x768)[0, 0, index]                     # values=tensor([[some high number]])
            grads = torch.autograd.grad(lx197x1000, lx197x768)
            grads = grads[0] # 1 197 768 All zero, except first row. First dim is tuple
            gradMatMul = torch.matmul(grads, lx197x768.transpose(2,1))
            gelu = nn.GELU()
            geluGrad = gelu(gradMatMul)    #gradMatMul | With Gelu and Without Gelu
            geluGrad14x14 = geluGrad[0][0][:-1].reshape(14, 14)
            cam = geluGrad14x14 - geluGrad14x14.min()
            cam = cam / cam.max()
            self.result = skimage.transform.resize(cam.detach().numpy(), [inputs.shape[-2], inputs.shape[-1]])

        else:
            fourLayers = nn.Sequential(*list(self.model.children())[:-2])(inputs) # size [1, 2048, 7, 7]
            classification = nn.Sequential(
                *(list(self.model.children())[-2:-1] + [Flatten()] + list(self.model.children())[-1:]))
            # classification() need for autograd
            grads = torch.autograd.grad(classification(fourLayers)[0, index], fourLayers)
            gradcam = torch.matmul(grads[0][0].mean(-1).mean(-1), fourLayers.view(fourLayers.shape[-3], fourLayers.shape[-2] * fourLayers.shape[-1])) # 1, 49
            gradcam = gradcam.view(fourLayers.shape[-2], fourLayers.shape[-1]).cpu().detach().numpy() # size: 7 7, np.ndarray
            gradcam = np.maximum(gradcam, 0)
            self.result = skimage.transform.resize(gradcam, [inputs.shape[-2], inputs.shape[-1]])
