import torch
import numpy as np

from .model import tokenizer, model

device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()

try:
    model.load_state_dict(torch.load('ppo/proofread/checkpoint.ckpt', map_location=device))
except FileNotFoundError:
    try:
        model.load_state_dict(torch.load('proofread/checkpoint.ckpt', map_location=device))
    except FileNotFoundError:
        print("failed to load checkpoint")


model.eval()


def predict(text: str) -> float:
    text = text.lower()  # preprocessing
    encoding = tokenizer(text, return_tensors='pt')
    out = model(**encoding)[0]
    out = torch.sigmoid(out)
    return float(out.cpu().detach().numpy()[0][0])

