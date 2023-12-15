import torch
import numpy as np
from torch import nn
from data.Resizer import Resizer


# Method create prediction for specifying list 
def predictionForList(model: torch.nn.Module, preprocessingList: list) -> list:
    return [model(pred) for pred in preprocessingList]


def gradCamMethod(model: torch.nn.Module, indexNumpy: list, preprocessingList: list) -> np.ndarray:
    # All layers before FNN
    layersBeforeClassif = nn.Sequential(*list(model.children())[:-2])
    # Adapt average and FNN layers without all other layers
    twoLastLayers = nn.Sequential(*(list(model.children())[-2:-1] \
                                                        + [Resizer()] \
                                                            + list(model.children())[-1:]))
    # activision map for all layers before 2 last
    l_2048x7x7 = [layersBeforeClassif(prep) for prep in preprocessingList]
    _, N, H, W = l_2048x7x7[0].size()  # 1 2048 7 7 map of saliency

    # insert in 2 last layers activision map 1 2048 7 7
    l_1x1000 = [twoLastLayers(l_2048x7x7[i])[0, indexNumpy[i]]
                                        for i in range(len(l_2048x7x7))]
    # print(l_2048x7x7[0], '\n', l_1x1000[0], l_2048x7x7[0].is_leaf, l_1x1000[0].is_leaf)


    grads = [torch.autograd.grad(l_1x1000[i], l_2048x7x7[i])
             for i in range(len(l_1x1000))]

    gradcam = [np.maximum(torch.matmul(grads[i][0][0].mean(-1).mean(-1), \
                                                        l_2048x7x7[i].view(N, H * W)) \
                                                            .view(H, W).cpu().detach().numpy(), 0)
               for i in range(len(l_1x1000))]
    return np.array(gradcam)