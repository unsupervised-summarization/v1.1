import torch
import numpy as np

from .model import tokenizer, model
from .data import DataLoader

data_loader = DataLoader(tokenizer)

device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()

try:
    model.load_state_dict(torch.load('ppo/proofread/checkpoint.ckpt', map_location=device))
except FileNotFoundError:
    model.load_state_dict(torch.load('proofread/checkpoint.ckpt', map_location=device))


try:
    raise Exception
    import torch_xla
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    xm.send_cpu_data_to_device(model, device())
    print('Using TPU on proofread')
except Exception:
    pass

model.eval()


def predict(text: str) -> float:
    text = data_loader.preprocessing(text)
    encoding = tokenizer(text, return_tensors='pt')
    out = model(**encoding)[0]
    out = torch.sigmoid(out)
    return float(out.cpu().detach().numpy()[0][0])

