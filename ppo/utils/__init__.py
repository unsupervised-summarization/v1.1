import torch
import numpy as np
from typing import List, Dict, Union


def wrap_into_batch(samples: List[torch.Tensor], batch_size: (int, int), dtype=None) -> (List[torch.Tensor], List[int]):
    # wrap samples that have a same shape.
    # ex) ([[1,2,3], [4,5], [6,7,8], [9]]) -> ([[[1,2,3],[6,7,8]], [[4,5]], [[9]]], [[0,2], [3], [4]])
    r: Dict[tuple, List[torch.Tensor]] = {}
    idx_dict: Dict[tuple, List[int]] = {}

    for i, sample in enumerate(samples):
        if sample.shape in r:
            r[sample.shape].append(sample.cpu().detach().numpy())
            idx_dict[sample.shape].append(i)
        else:
            r[sample.shape] = [sample.cpu().detach().numpy()]
            idx_dict[sample.shape] = [i]

    result: List[torch.Tensor] = []
    idx_result: List[List[int]] = []
    for v, idx in zip(r.values(), idx_dict.values()):
        try:
            size = v[0].numel()
        except:
            try:
                size = v[0].size
            except:
                size = len(v[0])
        if size <= 200:
            current_batch_size = batch_size[0]
        elif size <= 300:
            current_batch_size = batch_size[1]
        elif size <= 500:
            current_batch_size = batch_size[2]
        elif size <= 800:
            current_batch_size = batch_size[3]
        else:
            current_batch_size = batch_size[4]
        tmp = split_by_batch(torch.tensor(v, dtype=dtype), current_batch_size)
        result.extend(tmp)
        idx_result.extend(split_by_batch(idx, current_batch_size))
    return result, idx_result


def split_by_batch(samples: Union[List, torch.Tensor], batch_size: int) -> List[Union[List, torch.Tensor]]:
    # split samples by batch size
    # ex) f([1,2,3,4,5,6,7,8], 3) -> [[1,2,3], [4,5,6], [7,8]]
    return [samples[i:i+batch_size] for i in range(0, len(samples), batch_size)]


def fancy(lst: List, indexes: List[int], fn=None) -> List:
    # fancy indexing for list
    if fn is None:
        r = [lst[idx] for idx in indexes]
    else:
        r = [fn(lst[idx]) for idx in indexes]
    return r


def tensor_to_list(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy().tolist()


def to_numpy(tensor: torch.tensor) -> np.ndarray:
    # tensor to numpy
    return tensor.cpu().detach().numpy()


def shuffle(x: list) -> list:
    np.random.shuffle(x)
    return x
