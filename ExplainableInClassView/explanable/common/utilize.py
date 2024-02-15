from PIL import Image
import numpy as np
from skimage.transform import resize
from torch import nn
from explanable.log.LoggerModule import LoggerModuleClass

log = LoggerModuleClass()

def readImageAndPreprocessing(IMG_PATH: str, target_size: int = 224) -> np.ndarray:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if isinstance(IMG_PATH, str):
        with open(IMG_PATH, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = np.array(img)
            img = resize(img, (target_size, target_size))
            img = np.expand_dims(img, 0)
            img = img.astype('float32').transpose((0, 3, 1, 2))
            img_mean = np.array(mean).reshape((3, 1, 1))
            img_std = np.array(std).reshape((3, 1, 1))
            img -= img_mean
            img /= img_std
            log(f"img.shape: {img.shape}, img.type: {type(img)}")
        return img  # return [1,c,h,w]
    elif isinstance(IMG_PATH, np.ndarray):
        assert len(IMG_PATH.shape) == 4
        return IMG_PATH
    else:
        ValueError(f"Not recognized data type {type(IMG_PATH)}.")


def deprocessingImage(inputs: np.ndarray) -> np.ndarray:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    inputs *= img_std
    inputs += img_mean
    inputs *= 255
    inputs += 0.5  # snatch ???for float to integer
    if len(inputs.shape) == 4:
        img = np.uint8(inputs.transpose((0, 2, 3, 1)))  # [b h w channel]
    else:
        img = np.uint8(np.expand_dims(inputs).transpose((0, 2, 3, 1)))  # [b h w channel]
    return img

# service class for reversal dimension after pass four layers
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

class VitTransformerChanger(nn.Module):
    def __init__(self):
        super(VitTransformerChanger, self).__init__()
    def forward(self, x):
        # log(x.shape)
        return x['last_hidden_state']
