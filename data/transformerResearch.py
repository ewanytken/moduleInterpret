import torchvision
import numpy as np
import torch.nn as nn
import torch
import skimage.transform
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import topk

# Visualization and returned result are contained in same methods

# Need for retrieve from turple 'last_hidden_state'
class Changer(nn.Module):
    def __init__(self):
        super(Changer, self).__init__()
    def forward(self, x):
        return x['last_hidden_state']

# Restore from regular dimension 197x768 to 3x224x224. Method need for visual mapping
def restoreFromPatches224(patches: np.ndarray, kernelSize: int = 16, stride: int = 16) -> None:
    #input array: [batch] x [196or197] x [768] , 197th is position embedding, settled in last row
    _, x, _ = patches.size()
    if x == 197:
        patches = patches[:, :-1, :]
    patchReshape = patches.reshape(1, 14, 14, 3, kernelSize, stride)  #torch.Size([1, 3, 14, 14, 16, 16])
    patchReshape = patchReshape.permute(0, 3, 1, 2, 4, 5)             #torch.Size([1, 3, 14, 14, 16, 16])
    # batch, c, h, w
    unfoldShape = patchReshape.size()
    patchesOriginal = patchReshape.view(unfoldShape)
    outputH = unfoldShape[2] * unfoldShape[4] # 14 x 16 = 224
    outputW = unfoldShape[3] * unfoldShape[5] # 14 x 16 = 224
    patchesOriginal = patchesOriginal.permute(0, 1, 2, 4, 3, 5).contiguous()
    patchesOriginal = patchesOriginal.view(1, 3, outputH, outputW).squeeze(0)
    norm = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return norm(patchesOriginal)

# Similar method like gradCam. It is Reckoned production of weights from NormLayer and LastLayer
def gradForLayerNormLastLayer(model: torch.nn.Module, sample: torch.Tensor, rawImage) -> torch.Tensor:
    index = topk(model(sample)['logits'], 1)
    lastLayers = nn.Sequential(*list(model.children())[-1:])                   # ONLY LAST LAYER torch.Size([1, 197, 1000])
    beforeLastLayer = nn.Sequential(*list(model.children())[:-1] + [Changer()]) # torch.Size([1, 197, 768])
    lx197x768 = beforeLastLayer(sample)
    lx197x1000 = lastLayers(lx197x768)[0, 0, index.indices]                     # values=tensor([[some high number]])
    grads = torch.autograd.grad(lx197x1000, lx197x768)
    grads = grads[0] # 1 197 768 All zero, except first row. First dim is tuple
    gradMatMul = torch.matmul(grads, lx197x768.transpose(2,1))
    gelu = nn.GELU()
    geluGrad = gelu(gradMatMul)    #gradMatMul | With Gelu and Without Gelu
    geluGrad14x14 = geluGrad[0][0][:-1].reshape(14, 14)
    cam = geluGrad14x14 - geluGrad14x14.min()
    cam = cam / cam.max()
    plt.imshow(skimage.transform.resize(cam.detach().numpy(), [224, 224]), alpha=0.7, cmap='jet')
    plt.imshow(transforms.Compose([transforms.Resize((224, 224))])(rawImage), alpha=0.6)
    return grads

# Forward propagation picture through MHA
def weightsAfterPassFromViTAttention(model: nn.Module, sample: torch.Tensor, rawImage) -> list:
    fig = plt.figure(figsize=(48, 48))
    j = 0
    gelu = nn.GELU()
    embImage = model._modules.get('vit').get_submodule('embeddings')(sample)
    moduleAttAfterPass = []
    for i in range(8, 207, 18):
        j += 1
        fig.add_subplot(3, 4, j)
        viTLayer = list(model.modules())[i](embImage)[0]
        viTLayer = gelu(viTLayer)
        plt.imshow(transforms.Compose([transforms.Resize((224, 224))])(rawImage), alpha=0.6)
        plt.imshow(restoreFromPatches224(viTLayer).detach().numpy().squeeze().transpose(1,2,0), alpha=0.8, cmap='jet')
        moduleAttAfterPass.insert(j-1, viTLayer)
    return moduleAttAfterPass

# Backward propagation picture through MHA
def gradForViTAttentionLastLayer(model: nn.Module, sample: torch.Tensor, rawImage) -> list:
    fig = plt.figure(figsize=(48, 48))
    j = 0
    attentionVitLayer = []
    embImage = model._modules.get('vit').get_submodule('embeddings')(sample)
    lastLayers = nn.Sequential(*list(model.children())[-1:])                   # ONLY LAST LAYER torch.Size([1, 197, 1000])
    index = topk(model(sample)['logits'], 1)

    for i in range(8, 207, 18):
        j += 1
        fig.add_subplot(3, 4, j)
        viTLayer = list(model.modules())[i](embImage)
        viTLayer = viTLayer[0]
        lastTranspose = lastLayers(viTLayer)[0, 0, index.indices]  # values=tensor([[9.9484]])
        grads = torch.autograd.grad(lastTranspose, viTLayer)
        grads = grads[0]
        gradMatMul = torch.matmul(grads, viTLayer.transpose(2, 1))
        gelu = nn.ReLU()
        reluGrad = gelu(gradMatMul)  # gradMatMul |With Relu and Without Relu or GELU
        reluGrad = reluGrad[0][0][:-1].reshape(14, 14)
        cam = reluGrad  # - reluGrad.min()
        # cam = cam / cam.max()
        plt.imshow(skimage.transform.resize(cam.detach().numpy(), [224, 224]), alpha=0.7, cmap='jet')
        plt.imshow(transforms.Compose([transforms.Resize((224, 224))])(rawImage), alpha=0.6)
        attentionVitLayer.insert(j - 1, cam)
    return attentionVitLayer

# Similar method like CAM. It is Reckoned production of weights from NormLayer and LastLayer
def attentionCamSimilar(model: nn.Module, attention: list, sample: torch.Tensor, rawImage) -> list:
    fig = plt.figure(figsize=(48, 48))
    index = torch.squeeze(topk(model(sample)['logits'], 1)[1]).item()
    seqThreeLayers = nn.Sequential(*(list(model.children())[:-1] + [Changer()] + [list(model.children())[-1]])) # All layers by layers
    lastLayer = seqThreeLayers(sample) # torch.Size([1, 197, 1000])
    listOfAttention = []
    lastLayer = lastLayer.squeeze(0).T
    # index to 1000
    for i in range(len(attention)):
        mulWeightAtt = torch.matmul(lastLayer[index], attention[i])
        cam = mulWeightAtt - torch.min(mulWeightAtt)
        cam = cam / torch.max(cam)
        listOfAttention.insert(i, cam)
        fig.add_subplot(3, 4, i + 1)
        plt.imshow(transforms.Compose([transforms.Resize((224, 224))])(rawImage), alpha=0.6)
        plt.imshow(skimage.transform.resize(cam.reshape(3, 16, 16).permute(1,2,0).detach().numpy(), [224, 224]), alpha=0.8, cmap='jet')
    return listOfAttention