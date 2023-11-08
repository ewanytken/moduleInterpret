import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.autograd import Variable
from PIL import Image
from skimage.transform import resize
import os
from data.camMethods import imageMapping
from data.camMethods import preprocessingImage
from data.camMethods import visualizationCAM
from data.Hook import maxActivisionValue

from data.gradcamMethods import gradCamMethod
from data.gradcamMethods import predictionForList
