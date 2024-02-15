from torchvision import models
import torch
from explanable.common.utilize import readImageAndPreprocessing
from explanable.explainmethods import GuidedGramExpl
from explanable.visualization import visual
from explanable.metrics import MorfLerf
from explanable.visualization.visual import showChapter

if __name__ == '__main__':

    PATH_TO_IMAGE = "../image/deer.png"
    # PATH_TO_IMAGE = '../image/cat.jpg'
    inputs = readImageAndPreprocessing(PATH_TO_IMAGE)
    print(inputs.shape)

    model = models.resnet50()
    model.load_state_dict(torch.load('../pretrainingmodel/resnet50-0676ba61.pth'))
    model.eval()
    print()

    sm = GuidedGramExpl.GuidedGramExplClass(model)
    sm.explain(inputs)

    visual.showExplanation(sm.result, PATH_TO_IMAGE)

    ml = MorfLerf.MorfLerfClass(model)
    ml.setExplanation(sm)
    probeMorf, probeLerf = ml.evaluate(inputs)

    showChapter(probeMorf, probeLerf)


