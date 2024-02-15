import torch
from torch.nn import functional as F
import numpy as np
import transformers
# return one probability by certain index
def predictClassByDefineIndex(model: torch.nn.Module, inputs: np.ndarray, index: int) -> np.ndarray:
    with torch.no_grad():
        assert len(inputs.shape) == 4, 'Not enough dimension, are given less then 4 (model(*inputs)) '
        with torch.no_grad():
            if isinstance(inputs, np.ndarray):
                inputs = tuple(torch.tensor(inp) for inp in inputs) if isinstance(inputs, tuple) \
                    else (torch.tensor(inputs),)
            else:
                inputs = tuple(inp for inp in inputs) if isinstance(inputs, tuple) \
                    else (inputs,)

        if type(model) == transformers.models.vit.modeling_vit.ViTForImageClassification:
            probas = F.softmax(model(*inputs).logits, dim=1)  # get probabilities
            classProba = probas[:, index]
        else:
            probas = F.softmax(model(*inputs) , dim=1)  # get probabilities
            classProba = probas[:, index]

    return np.array(classProba, dtype=np.float64)

# return class probability and current index
def predictClassWithIndex(model: torch.nn.Module, inputs: np.ndarray):
    with torch.no_grad():
        assert len(inputs.shape) == 4, 'Not enough dimension, are given less then 4 (model(*inputs)) '
        with torch.no_grad():
            if isinstance(inputs, np.ndarray):
                inputs = tuple(torch.tensor(inp) for inp in inputs) if isinstance(inputs, tuple) \
                    else (torch.tensor(inputs),)
            else:
                inputs = tuple(inp for inp in inputs) if isinstance(inputs, tuple) \
                    else (inputs,)

        if type(model) == transformers.models.vit.modeling_vit.ViTForImageClassification:
            probas = F.softmax(model(*inputs).logits, dim=1)  # get probabilities
            index = np.argmax(probas[0].detach().numpy(), axis=0)
            classProba = probas[:, index]
        else:
            probas = F.softmax(model(*inputs), dim=1)  # get probabilities
            index = np.argmax(probas[0], axis=0)
            classProba = probas[:, index]
    return np.array(classProba, dtype=np.float64), index


