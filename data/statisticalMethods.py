import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights
from torch.autograd import Variable
from skimage.io import imread
from PIL import Image
from skimage.transform import resize
import os
import math
from scipy.stats import logistic, uniform, norm, pearsonr
from scipy import stats
from data.Hook import Hook

def showStatisticalVis(path: str, features: torch.tensor, col : int =2) -> None:
    row = math.ceil(features.shape[0] / col)
    fig = plt.figure(figsize=(7, 85))
    listDir = os.listdir(path)
    for pcs in range(features.shape[0]):
        fig.add_subplot(row, col, pcs+1)
        plt.hist(features[pcs], bins='auto', density=True)
        plt.plot(np.sort(np.array(features[pcs])), norm.pdf(np.sort(np.array(features[pcs])),\
                          features[pcs].mean(),\
                          features[pcs].std())) 
        
        plt.plot(np.sort(np.array(features[pcs])), logistic.pdf(np.sort(np.array(features[pcs])),\
                          features[pcs].mean(),\
                          features[pcs].std())) 
        
        plt.title(
                   listDir[pcs] + \
                  '\nNorm statistic: '+ str(stats.kstest(np.sort(np.array(features[pcs])), \
                                                    norm.cdf(np.sort(np.array(features[pcs])),\
                                                      features[pcs].mean(),\
                                                      features[pcs].std())).statistic) +
                  
                '\nLog statistic: '+ str(stats.kstest(np.sort(np.array(features[pcs])), \
                                  logistic.cdf(np.sort(np.array(features[pcs])),\
                                  features[pcs].mean(),\
                                  features[pcs].std())).statistic),
                 fontsize=7)
    
        plt.grid('on')
        plt.axis('on')
        

# Need Hook-class
def statisticalDataFromLayers(model: torch.nn.Module, preprocessingList: list,
                   layersToResearch: list) -> torch.Tensor:
    resultListOfTensors = []
    for i in range(len(layersToResearch)):
        layer = layersToResearch[i]
        actLayer = Hook(model._modules.get(layer)) # Get the intermediate result of the entire Lenet model conv2
        [model(pred) for pred in preprocessingList]
        actLayer.remove()
        featuresToTensor = torch.stack(actLayer.features).squeeze(1)
        print(layer)
        if layersToResearch[i] == 'fc':
            # fc consist of len of pcs, len FNN
            tensorSqueezeByMean = layersToResearch
        else: 
            # len of pcs, channel, w, h -> len_of_pcs, w*h; channel squeeze by mean function
            tensorSqueezeByMean = featuresToTensor.reshape(featuresToTensor.size(0), featuresToTensor.size(1), 
                                 featuresToTensor.size(2)*featuresToTensor.size(3)) \
                                        .mean(axis=1) 
        actLayer.clearFeatures()
        resultListOfTensors.insert(i, tensorSqueezeByMean)
        
    return resultListOfTensors
