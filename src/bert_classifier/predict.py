'''
Inference module
'''
import torch
from .bert import bert_encoder
from .io import get_pretrained_tokenizer


MAX_LEN = 512


class Inference:
    '''
    inference class
    '''
    def __init__(self, model, labels, tokenizer=None, max_len=MAX_LEN):
        self.tokenizer = tokenizer if tokenizer else get_pretrained_tokenizer()
        self.model = model
        self.labels = labels
        self.max_len = max_len

    def predict(self, inp: str):
        '''
        inp: text input
        '''
        self.model.eval()
        enc = bert_encoder(inp, self.tokenizer, self.max_len)
        out = self.model(**enc)[-1].detach().cpu()
        idx = out.argmax().item()
        label = self.labels[idx]
        proba = out.softmax(-1)[idx].item()
        return label, proba

    def __call__(self, inp):
        return self.predict(inp)


def load_model(model_dir, device='cpu'):
    '''
    load full model
    '''
    return torch.load(model_dir, map_location=device)
