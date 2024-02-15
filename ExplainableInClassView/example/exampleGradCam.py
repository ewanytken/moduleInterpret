from torchvision import models
import torch
from transformers import ViTForImageClassification

from explanable.common.utilize import readImageAndPreprocessing
from explanable.explainmethods.GradCamExpl import GradCamExplClass
from explanable.visualization import visual
from explanable.metrics.MorfLerf import MorfLerfClass
from explanable.visualization.visual import showChapter

if __name__ == '__main__':

    # PATH_TO_IMAGE = "../image/deer.png"
    PATH_TO_IMAGE = '../image/cat.jpg'
    inputs = readImageAndPreprocessing(PATH_TO_IMAGE)

    print(inputs.shape)

    # model = models.resnet50()
    # model.load_state_dict(torch.load('../pretrainingmodel/resnet50-0676ba61.pth'))
    # model.eval()
    # print()

    # gc = GradCamExplClass(model)
    # gc.explain(inputs)
    # print(gc.result.shape)
    #
    # visual.showExplanation(gc.result, PATH_TO_IMAGE)
    #
    # ml = MorfLerfClass(model)
    # ml.setExplanation(gc)
    # probeMorf, probeLerf = ml.evaluate(inputs)
    #
    # showChapter(probeMorf, probeLerf)

    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    gcVit = GradCamExplClass(model)
    gcVit.explain(inputs)
    visual.showExplanation(gcVit.result, PATH_TO_IMAGE)

    ml = MorfLerfClass(model)
    ml.setExplanation(gcVit)
    probeMorf, probeLerf = ml.evaluate(inputs)

    showChapter(probeMorf, probeLerf)

