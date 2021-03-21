
from fastai.basics import *

class RModel(Module):
    def __init__(self):
        super(RModel, self).__init__()

    def forward(self, *args):
      return self._r_call(*args)

