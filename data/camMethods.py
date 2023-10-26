from torchvision import transforms, models
import torch
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import skimage.transform
import math
import os

# pics_mapping return list of Image and show images in set
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

# regular preprocessing for Image, return torch.tensor ([1, 3, 224, 224])
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

# Methods creates CAM for one pcs and for set of CAM
def camMethod(features, weight, index):
    features = features.cpu().data.numpy()
    _, nc, h, w = features.shape
    imp = weight[index].dot(features.reshape((nc, h*w))).reshape(h, w)
    cam = imp - np.min(imp)
    cam = cam / np.max(cam)
    return cam

# Cam for set
def camForSet(actFeatures: list, weight: np.ndarray, index: list) -> list:
    return [camMethod(act, weight, idx)
            for act, idx in zip(actFeatures, index)]

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

