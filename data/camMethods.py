from torchvision import transforms, models
import torch
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk #return max element in dim (can return K-max elements)
import numpy as np
import skimage.transform
import math
import os
from sklearn.decomposition import PCA
from data.Hook import actLayerMethod
from data.Hook import maxActivisionValue

# Pics_mapping return list of Image and show all images in set
def imageMapping(path: str, col: int = 3) -> list:
    listDir = os.listdir(path)
    listImage = []
    row = math.ceil(len(listDir)/3)
    fig = plt.figure(figsize=(12, 12))
    for pcs in range(len(listDir)):
        fig.add_subplot(row, col, pcs+1)
        listImage.append(Image.open(os.path.join(path, str(listDir[pcs]))))
        plt.imshow(Image.open(os.path.join(path, str(listDir[pcs]))))
        plt.title(label=listDir[pcs], fontsize=5)
        plt.axis('off')
    return listImage

# Regular preprocessing for Image, return torch.tensor ([1, 3, 224, 224])
def preprocessingImage(listImage: list) -> list:
    preprocessingCallable = transforms.Compose([

        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        transforms.Compose([transforms.Resize((224, 224))])
    ])
    return [Variable((preprocessingCallable(image).unsqueeze(0)), requires_grad=False) \
            for image in listImage]

# Method create CAM for one pcs and for set of CAM
def camMethod(features, weight, index, modeFC = True):
    features = features.cpu().data.numpy()
    _, nc, h, w = features.shape
    if modeFC == True: 
        imp = weight[index].dot(features.reshape((nc, h*w))).reshape(h, w)
    else: 
        imp = weight.dot(features.reshape((nc, h*w))).reshape(h, w) 
        
    cam = imp - np.min(imp)
    cam = cam / np.max(cam)
    return cam

# Cam for set
def camForSet(actFeatures: list, weight: np.ndarray, index: list, modeFC : bool = True) -> np.ndarray:
    return np.array([camMethod(act, weight, idx, modeFC) #change return type, do we get add dim?
            for act, idx in zip(actFeatures, index)])

# Visualization for CAM set
def visualizationCAM(path: str, camForPictures: np.ndarray, col: int = 2) -> None:
    listDir = os.listdir(path)
    row = math.ceil(len(listDir) / col)
    fig = plt.figure(figsize=(128, 128))
    display = transforms.Compose([transforms.Resize((224, 224))])

    for pcs in range(len(listDir)):
        fig.add_subplot(row, col, pcs + 1)
        plt.imshow(display(Image.open(os.path.join(path, str(listDir[pcs])))))
        plt.imshow(skimage.transform.resize(camForPictures[pcs],
                                            [224, 224]),
                   alpha=0.7,
                   cmap='jet')

        plt.title(label=listDir[pcs], fontsize=5)
        plt.axis('off')

# Method obtain simulitary and original cam
def camsForLayersRes50(model: torch.nn.Module, preprocessingList: list,
                                listOfWeight: list, startLayer: int = 1) -> list:
    mapping = []
    for i in range(listOfWeight.shape[0]):
        modeFC = False # mode for last layers FC and layer4's cam. See implementation of actLayerMethod
        weightLayer = np.squeeze(list(listOfWeight[i].parameters())[0]\
                                                        .cpu().data.numpy())
        
        layer = 'layer{}'.format(i+startLayer)      # start from first layer
        idxNumpy = np.empty(len(preprocessingList)) # stub for speed-up, Cam-method have idx for all case

        if listOfWeight[i] == model._modules.get('fc'):
            layer = 'layer4' #map activisition for fc layer
            idxNumpy = maxActivisionValue(predictionList)
            modeFC = True
        features, predictionList = actLayerMethod(model, preprocessingList,
                                                                      layer)
#         print(i, features[0].cpu().data.numpy().shape, idxNumpy.shape, modeFC ) # control value
        mapping.insert(i, camForSet(features, weightLayer, 
                                    idxNumpy, modeFC))
        
    return mapping