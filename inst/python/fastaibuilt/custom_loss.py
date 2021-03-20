
import torch.nn as nn
#https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch

class CustomLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CustomLoss, self).__init__()

    def forward(self, **kwargs):
        return self._r_call(**kwargs)

