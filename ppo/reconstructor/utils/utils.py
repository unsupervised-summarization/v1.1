from typing import Dict, List, Union
import torch
import numpy as np
from ..args import args


def set_device(*tensors, device=None) -> Union[List[torch.tensor], torch.tensor]:
    # set tensors to device
    if device is None:
        device = args['device']

    if len(tensors) == 1:
        return tensors[0].to(device)
    return [t.to(device) for t in tensors]


def to_numpy(tensor: torch.tensor) -> np.ndarray:
    # tensor to numpy
    return tensor.cpu().detach().numpy()


def shuffle(x: list) -> list:
    np.random.shuffle(x)
    return x
