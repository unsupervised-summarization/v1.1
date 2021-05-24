from ..model import load_tokenizer
from ..data import DataLoader
from .reward import Reward

import torch
from typing import List


class Env:
    def __init__(self):
        # tokenizer
        self.tokenizer = load_tokenizer()
        self.eos_token_id = self.tokenizer.eos_token_id

        self.action_dim = self.tokenizer.vocab_size  # policy model output dim

        self.data_loader = DataLoader()
        self.actions: List[int] = []  # list of actions
        self.document: str = ''
        self.init_state: List[int] = []
        self.max_len = 100  # max length of actions (== length of summary)

        self.reward_class = Reward()

    def reset(self, document: str = '') -> torch.LongTensor:
        # Rest the environment
        # return state
        self.reward_class = Reward()

        self.actions: List[str] = []
        if document == '':
            n = 9999
            while n >= 1023 - self.max_len:  # number of tokens should be 1024 or less
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
        summary: str = self.tokenizer.decode(self.actions)
        reward = self.reward_class.get_reward(self.document, summary, log=log)
        return reward

    def close(self):
        # Close the environment
        pass
