from ..model import load_tokenizer
from ..data import DataLoader
from .reward import Reward
from ..reconstructor.utils import Logger

import pickle
import torch
from typing import List, Tuple


class Env:
    def __init__(self):
        # tokenizer
        self.tokenizer = load_tokenizer()
        self.eos_token_id = self.tokenizer.eos_token_id

        self.action_dim = self.tokenizer.vocab_size  # policy model output dim

        self.data_loader = DataLoader()
        self.actions: List[int] = []  # list of actions
        self.document: str = ''
        self.summary: str = ''
        self.init_state: List[int] = []  # initial state
        self.max_len = 100  # max length of actions (== length of summary)

        self.history: List[Tuple[str, str]] = []  # (document, summary)

        self.reward_class = Reward()
        self.reward_logger = Logger()
        self.reward_logger_tmp: List[float] = []  # stack rewards until updating, then append it into logger.

    def reset(self, document: str = '') -> torch.LongTensor:
        # Rest the environment
        # return state

        self.summary: str = ''
        self.actions: List[str] = []
        if document == '':
            n = 9999
            while n >= 1000 - self.max_len:  # number of tokens should be 1024 or less
                self.document: str = self.data_loader.get_n(1)[0]
                outputs = self.tokenizer(self.document)
                n = len(outputs['input_ids'])
        else:
            self.document: str = document
            outputs = self.tokenizer(self.document)
        self.init_state = outputs['input_ids']

        state = torch.LongTensor(self.init_state)
        return state

    def step(self, action: int) -> (torch.LongTensor, bool):
        # return (state, attention_mask, done)
        # action == token_id
        self.actions.append(action)

        state = self.get_state()
        done = (action == self.eos_token_id)
        if self.max_len == len(self.actions):
            done = True

        return state, done

    def get_state(self) -> torch.LongTensor:
        # return state
        state = self.init_state + [self.tokenizer.bos_token_id] + self.actions
        state = torch.LongTensor(state)
        return state

    def get_reward(self, log: bool = False) -> float:
        # Calculate reward
        # return reward
        self.summary: str = self.tokenizer.decode(self.actions)
        self.summary = self.summary.replace('<|endoftext|>', '')
        self.history.append((self.document, self.summary))
        reward = self.reward_class.get_reward(self.document, self.summary, log=log)

        self.reward_logger_tmp.append(reward)  # logging a reward

        return reward

    def update_reward_model(self, reconstructor_lr: float = None, epochs: int = 5, batch_size: int = 4):
        # train reward model
        documents, summaries = list(zip(*self.history))

        # reconstructor
        self.reward_class.train_reconstructor(
            documents, summaries, lr=reconstructor_lr, epochs=epochs, batch_size=batch_size
        )

        self.history = []

    def append_log(self):
        print(len(self.reward_logger.logs))
        self.reward_logger(self.reward_logger_tmp.copy())  # logging stacked rewards
        self.reward_logger_tmp = []
        self.reward_class.append_logger()

    def close(self):
        # Close the environment
        pass

    def save(self, path: str):
        # save
        with open(path, 'wb') as f:
            pickle.dump((self.reward_class, self.reward_logger), f)

    def load(self, path: str):
        # load
        with open(path, 'rb') as f:
            self.reward_class, self.reward_logger = pickle.load(f)
