from torch import nn

#crutch -> squeeze all dimension, except first. Representation in list cannot reshape from 2048x1 to 1x2048.
class Resizer(nn.Module):
    def __init__(self):
        super(Resizer, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)