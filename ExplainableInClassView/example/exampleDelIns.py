from torchvision import models
import torch
from explanable.common.utilize import readImageAndPreprocessing
from explanable.explainmethods.GradCamExpl import GradCamExplClass
from explanable.visualization import visual
from explanable.metrics.DeletionInsertion import DeletionInsertionClass
from explanable.visualization.visual import showChapter
from explanable.explainmethods.CamExpl import CamExplClass

if __name__ == '__main__':

    # PATH_TO_IMAGE = "../image/deer.png"
    PATH_TO_IMAGE = '../image/cat.jpg'
    inputs = readImageAndPreprocessing(PATH_TO_IMAGE)

    model = models.resnet50()
    model.load_state_dict(torch.load('../pretrainingmodel/resnet50-0676ba61.pth'))
    model.eval()
    print()

    gc = GradCamExplClass(model)
    gc.explain(inputs)

    cam = CamExplClass(model)
    cam.explain(inputs)

    visual.showExplanation(cam.result, PATH_TO_IMAGE)

    ml = DeletionInsertionClass(model)
    ml.setExplanation(gc)
    probeMorf, probeLerf = ml.evaluate(inputs)

    showChapter(probeMorf, probeLerf)

