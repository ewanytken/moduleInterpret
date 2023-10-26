import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import torch
import torch.nn as nn
from PIL import Image
from skimage.transform import resize
import os
import math
from scipy.stats import logistic, uniform, norm, pearsonr
from scipy import stats
from data.Hook import Hook

def show_hist(features: torch.tensor, col: int = 2) -> None:
    row = math.ceil(features.shape[0] / col)
    fig = plt.figure(figsize=(7, 85))
    for pcs in range(features.shape[0]):
        fig.add_subplot(row, col, pcs + 1)
        plt.hist(features[pcs], bins='auto', density=True)
        plt.plot(np.sort(np.array(features[pcs])), norm.pdf(np.sort(np.array(features[pcs])), \
                                                            features[pcs].mean(), \
                                                            features[pcs].std()))

        plt.plot(np.sort(np.array(features[pcs])), logistic.pdf(np.sort(np.array(features[pcs])), \
                                                                features[pcs].mean(), \
                                                                features[pcs].std()))

        plt.title(
            os.listdir()[pcs] + \
            '\nNorm statistic: ' + str(stats.kstest(np.sort(np.array(features[pcs])), \
                                                    norm.cdf(np.sort(np.array(features[pcs])), \
                                                             features[pcs].mean(), \
                                                             features[pcs].std())).statistic) +

            '\nLog statistic: ' + str(stats.kstest(np.sort(np.array(features[pcs])), \
                                                   logistic.cdf(np.sort(np.array(features[pcs])), \
                                                                features[pcs].mean(), \
                                                                features[pcs].std())).statistic),
            fontsize=7)

        plt.grid('on')
        plt.axis('on')

# Need Hook-class
def statistical_data_for_layers(model: torch.nn.Module, preproc_list: list, layers_to_search: list) -> None:
    for i in range(len(layers_to_search)):
        layer = layers_to_search[i]
        act_layer = Hook(model._modules.get(layer)) # Get the intermediate result of the entire Lenet model conv2
        [model(pred) for pred in preproc_list]
        act_layer.remove()
        features_to_tensor = torch.stack(act_layer.features).squeeze(1)
        if layers_to_search[i] == 'fc':
            # fc consist of len of pcs, len FNN
            mean_tensor = features_to_tensor
        else:
            # len of pcs, channel, w, h -> len_of_pcs, w*h; channel squeeze by mean function
            mean_tensor = features_to_tensor.reshape(features_to_tensor.size(0), features_to_tensor.size(1),
                                 features_to_tensor.size(2)*features_to_tensor.size(3)) \
                                        .mean(axis=1)
        show_hist(mean_tensor)
        act_layer.clear_features()