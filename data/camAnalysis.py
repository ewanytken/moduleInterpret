import math 
from sklearn.decomposition import PCA
import torch
import numpy as np

def shrinkVectors(vectors : np.ndarray, comp : int) -> None:
    pca = PCA(n_components = comp)
    afterPCA = pca.fit_transform(vectors)
    return np.array([afterPCA[i] \
            .reshape(int(math.sqrt(comp)), int(math.sqrt(comp))) \
               for i in range(afterPCA.shape[0])])        

def vectorFromCam(features, weight, index, modeFC = True):
    features = features.cpu().data.numpy()
    _, nc, h, w = features.shape
    if modeFC == True: 
        imp = weight[index].dot(features.reshape((nc, h*w)))
    else: 
        imp = weight.dot(features.reshape((nc, h*w))) 
        
    cam = imp - np.min(imp)
    cam = cam / np.max(cam)
    return cam

# Simularly CAM Methods, but return vectors  
def vectorForSet(actFeatures: list, weight: np.ndarray, index: list, modeFC : bool = True) -> np.ndarray:
    return np.array([vectorFromCam(act, weight, idx, modeFC) 
            for act, idx in zip(actFeatures, index)])