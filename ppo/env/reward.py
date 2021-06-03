from typing import List, Union, Tuple
import torch
import numpy as np

from ..proofread import predict as proofread_predict
from ..repr_checker.checker.check import ReprChecker
from ..reconstructor.trainer import Trainer
from ..utils import to_numpy, shuffle
from ..utils.buffer import Buffer
from ..utils.batch_loader import batch_loader
from ..reconstructor.utils import Logger


class Reward:
    def __init__(self):
        self.checker = ReprChecker()
        try:
            self.checker.load('ppo/repr_checker/checkpoint.ckpt')
        except FileNotFoundError:
            self.checker.load('repr_checker/checkpoint.ckpt')

        self.reconstructor = Trainer()
        try:
            self.reconstructor.load('ppo/reconstructor/pretrained.ckpt')
        except FileNotFoundError:
            self.reconstructor.load('reconstructor/pretrained.ckpt')
        self.reconstructor_loss_buffer = Buffer(maxlen=10000)

        self.reward_names = ['repr_checker', 'proofread', 'length', 'reconstruct']
        self.reward_logger = Logger(*self.reward_names)
        self.reward_logger_tmp = {  # stack rewards until updating, then append it into logger.
            name: [] for name in self.reward_names
        }

    def append_logger(self):
        # append reward_logger_tmp into logger and reset reward_logger_tmp
        for name in self.reward_names:
            self.reward_logger[name](self.reward_logger_tmp[name].copy())  # logging stacked rewards
            self.reward_logger_tmp[name] = []

    def repr_check(self, document: str, summary: str) -> float:
        # output : 0~1
        return self.checker.check(document, summary, caching=True)

    def train_reconstructor(self, documents: List[str], summaries: List[str],
                            lr: float = None, epochs: int = 5, batch_size: int = 4):
        # train reconstructor
        documents_tokens = [self.reconstructor.tokenizer.encode(sent) for sent in documents]
        summaries_tokens = [self.reconstructor.tokenizer.encode(sent) for sent in summaries]
        for e in range(epochs):
            documents_tokens, summaries_tokens = zip(*shuffle(list(zip(documents_tokens, summaries_tokens))))  # shuffle
            for summaries_tokens_batch, documents_tokens_batch in \
                    batch_loader(summaries_tokens, documents_tokens, batch_size):
                self.reconstructor.train_step(
                    summaries_tokens_batch,
                    documents_tokens_batch,
                    train=True,
                    lr=lr,
                )

    def reconstruct_check(self, document: str, summary: str) -> float:
        documents_tokens = [self.reconstructor.tokenizer.encode(document)]
        summaries_tokens = [self.reconstructor.tokenizer.encode(summary)]
        with torch.no_grad():
            loss = self.reconstructor.train_step(
                summaries_tokens,
                documents_tokens,
                train=False
            )  # just return loss without training
        loss = to_numpy(loss)
        self.reconstructor_loss_buffer.append(loss)
        return -(loss / self.reconstructor_loss_buffer.percentile(80))  # WARNING: this reward is not 0~1 value.

    @staticmethod
    def proofread(summary: str) -> float:
        # output : 0~1
        return proofread_predict(summary)

    @staticmethod
    def length(document: str, summary: str) -> float:
        # output : -1~0

        doc_len: int = document.count(' ')
        sum_len: int = summary.count(' ')
        assert doc_len > 0

        ratio: float = (sum_len / doc_len)  # 0~1
        min_ratio = 0.2
        if ratio < min_ratio:
            ratio = min_ratio ** (19 * ratio)
        else:
            ratio = -(9 ** (-ratio + 0.45 * np.log(
                19073486328125.0 ** (-min_ratio) * (19073486328125.0 ** min_ratio - 1.0) * np.exp(
                    2.19722457733622 * min_ratio)))) + 1
        return -ratio  # -1~0

    @staticmethod
    def weighted_sum(inputs: Union[List[float], Tuple[float]], weights: Union[List[float], Tuple[float]]) -> float:
        # return weighted sum
        assert 0. not in weights
        inputs: np.ndarray = np.array(inputs)
        weights: np.ndarray = np.array(weights)

        weights = weights / weights.mean()  # normalize weights
        return float((inputs*weights).sum() / weights.sum())

    def get_reward(self, document: str, summary: str, log: bool = False) -> float:
        # return reward : -1 ~ 2
        weights = (4, 1, 1, 20)
        r = (
            self.repr_check(document, summary),
            self.proofread(summary),
            self.length(document, summary),
            self.reconstruct_check(document, summary),
        )
        alpha = 0
        if summary.count(' ') <= 5:
            alpha = -1
        if summary.count(' ') <= 1:
            alpha = -2
        if log:
            print(r, alpha)

        self.reward_logger_tmp['repr_checker'].append(r[0])
        self.reward_logger_tmp['proofread'].append(r[1])
        self.reward_logger_tmp['length'].append(r[2])
        self.reward_logger_tmp['reconstruct'].append(r[3])

        return self.weighted_sum(r, weights) + alpha
