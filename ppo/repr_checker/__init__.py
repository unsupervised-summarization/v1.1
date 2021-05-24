import torch
import numpy as np

from .model import tokenizer, model
from .data import DataLoader

data_loader = DataLoader(tokenizer)

device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()


model.eval()


def predict(text: str) -> float:
    text = data_loader.preprocessing(text)
    encoding = tokenizer(text, return_tensors='pt')
    out = model(**encoding)[0]
    out = torch.sigmoid(out)
    return float(out.cpu().detach().numpy()[0][0])

