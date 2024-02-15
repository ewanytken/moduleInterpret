from torchvision import models
import torch
from explanable.common.utilize import readImageAndPreprocessing
from explanable.explainmethods.RiseExpl import RiseExplClass
from explanable.visualization.visual import showExplanation
from explanable.metrics.DeletionInsertion import DeletionInsertionClass
from explanable.visualization.visual import showChapter
from explanable.visualization.visual import showByPictures

if __name__ == '__main__':

    # PATH_TO_IMAGE = "../image/deer.png"
    PATH_TO_IMAGE = '../image/cat.jpg'
    inputs = readImageAndPreprocessing(PATH_TO_IMAGE)

    model = models.resnet50()
    model.load_state_dict(torch.load('../pretrainingmodel/resnet50-0676ba61.pth'))
    model.eval()
    print()

    rec = RiseExplClass(model)
    rec.explain(inputs)

    showByPictures(rec.intermediaResults)

    showExplanation(rec.result, PATH_TO_IMAGE)

    delIns = DeletionInsertionClass(model)
    delIns.setExplanation(rec)
    probeMorf, probeLerf = delIns.evaluate(inputs)

    showChapter(probeMorf, probeLerf)