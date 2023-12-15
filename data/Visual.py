from torchvision import transforms, models
import torch
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import math
import os
from scipy.stats import logistic, uniform, norm, pearsonr
from scipy import stats

class Visual():
    def __init__(self, path: str, mapping, col: int = 2) -> None:
        self.path = path
        self.mapping = mapping
        self.col = col

    def visualization(self) -> None:
        listDir = os.listdir(self.path)
        row = math.ceil(len(listDir) / self.col)
        fig = plt.figure(figsize=(128, 128))
        display = transforms.Compose([transforms.Resize((224, 224))])

        for pcs in range(len(listDir)):
            fig.add_subplot(row, self.col, pcs + 1)
            plt.imshow(display(Image.open(os.path.join(self.path, str(listDir[pcs])))))
            plt.imshow(skimage.transform.resize(self.mapping[pcs],
                                                [224, 224]),
                       alpha=0.7,
                       cmap='jet')

            plt.title(label=listDir[pcs], fontsize=5)
            plt.axis('off')
            
    def visualization(self) -> None:
        listDir = os.listdir(self.path)
        row = math.ceil(self.mapping.shape[0] /self.col)
        fig = plt.figure(figsize=(7, 85))
        for pcs in range(self.mapping.shape[0]):
            fig.add_subplot(row, self.col, pcs+1)
            plt.hist(self.mapping[pcs], bins='auto', density=True)
            plt.plot(np.sort(np.array(self.mapping[pcs])), norm.pdf(np.sort(np.array(self.mapping[pcs])),\
                              self.mapping[pcs].mean(),\
                              self.mapping[pcs].std())) 

            plt.plot(np.sort(np.array(self.mapping[pcs])), logistic.pdf(np.sort(np.array(self.mapping[pcs])),\
                              self.mapping[pcs].mean(),\
                              self.mapping[pcs].std())) 

            plt.title(
                       listDir[pcs] + \
                      '\nNorm statistic: '+ str(stats.kstest(np.sort(np.array(self.mapping[pcs])), \
                                                        norm.cdf(np.sort(np.array(self.mapping[pcs])),\
                                                          self.mapping[pcs].mean(),\
                                                          self.mapping[pcs].std())).statistic) +

                    '\nLog statistic: '+ str(stats.kstest(np.sort(np.array(self.mapping[pcs])), \
                                      logistic.cdf(np.sort(np.array(self.mapping[pcs])),\
                                      self.mapping[pcs].mean(),\
                                      self.mapping[pcs].std())).statistic),
                     fontsize=7)

            plt.grid('on')
            plt.axis('on')