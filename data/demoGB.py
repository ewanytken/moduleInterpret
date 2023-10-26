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

