import torch
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from explanable.log.LoggerModule import LoggerModuleClass

log = LoggerModuleClass()

# entry ndarray size of [1,3,224,224]
def showExplanation(explanation, imgPath, targetSize=224) -> None:
    if isinstance(explanation, np.ndarray) and len(explanation.shape) == 4:
        explanation = explanation.squeeze(0).transpose(1, 2, 0)
    if isinstance(explanation, torch.Tensor) and len(explanation.shape) == 4:
        explanation = explanation.detach().numpy().squeeze(0).transpose(1, 2, 0)
    if isinstance(imgPath, str):
        with open(imgPath, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = np.array(img)
            img = resize(img, (targetSize, targetSize))
    imshow(explanation, cmap='jet')
    imshow(img, alpha=0.5)

def showByPictures(results: dict, showImage = 10) -> None:
    psize = 4
    cols = showImage
    if isinstance(results, dict):
        if showImage > len(list(results.values())[0]) \
                or showImage > len(list(results.values())[1]):
            showImage = len(list(results.values())[0])

        fig, ax = plt.subplots(1, cols, figsize=(cols * psize, 1 * psize))
        for axis, img in zip(ax, list(results.values())[0][:showImage]):
            axis.axis('off')
            axis.imshow(img.transpose(1, 2, 0))
        plt.show()

        fig, ax = plt.subplots(1, cols, figsize=(cols * psize, 1 * psize))
        for axis, img in zip(ax, list(results.values())[1][:showImage]):
            axis.axis('off')
            axis.imshow(img.transpose(1, 2, 0))
        plt.show()

    elif isinstance(results, np.ndarray):
        if showImage > results.shape[0]:
            showImage = results.shape[0]
        fig, ax = plt.subplots(1, cols, figsize=(cols * psize, 1 * psize))
        for axis, img in zip(ax, results[:showImage]):
            axis.axis('off')
            axis.imshow(img.transpose(1, 2, 0))
        plt.show()
    else:
        log(f'Method cannot work with this type: {type(results)}')

def showChapter(probeOne: np.ndarray, probeTwo: np.ndarray) -> None:
    if probeTwo[0] > probeTwo[-1]: # insertion raise value of probeTwo from step to step
        CHAPTER_ONE = 'MoRF Curve'
        CHAPTER_TWO = 'LeRF Curve'
        COLOR = 'blue'
    else:
        CHAPTER_ONE = 'Deletion'
        CHAPTER_TWO = 'Insertion'
        COLOR = 'red'

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(np.arange(len(probeOne)) / len(probeOne), probeOne, color=COLOR)
    axes[0].text(0.7, 0.7, f"AUC={np.mean(probeOne):.2f}")
    axes[0].fill_between(x=np.arange(len(probeOne)) / len(probeOne),
                         y1=0,
                         y2=probeOne,
                         alpha=0.2,
                         facecolor=COLOR)

    axes[0].set_ylim((0, 1.05))
    axes[0].set_xlabel('Pixels Removed Ratio')
    axes[0].set_ylabel('Probability')
    axes[0].set_title(CHAPTER_ONE)

    axes[1].plot(np.arange(len(probeTwo)) / len(probeTwo), probeTwo, color=COLOR)
    axes[1].text(0.7, 0.7, f"AUC={np.mean(probeTwo):.2f}")
    log(f'Probability 1: {np.mean(probeOne)} probability 2: {np.mean(probeTwo)}')
    axes[1].fill_between(x=np.arange(len(probeTwo)) / len(probeTwo),
                         y1=0,
                         y2=probeTwo,
                         alpha=0.2,
                         facecolor=COLOR)

    axes[1].set_ylim((0, 1.05))
    axes[1].set_xlabel('Pixels Removed Ratio')
    axes[1].set_ylabel('Probability')
    axes[1].set_title(CHAPTER_TWO)