

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from fastai.text.all import *

def tokenize(text):
    toks = tokenizer.tokenize(text)
    return tensor(tokenizer.convert_tokens_to_ids(toks))

class TransformersTokenizer(Transform):
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def encodes(self, x): 
        return x if isinstance(x, Tensor) else tokenize(x)
        
    def decodes(self, x): return TitledStr(self.tokenizer.decode(x.cpu().numpy()))

class TransformersDropOutput(Callback):
    def after_pred(self): self.learn.pred = self.pred[0]



