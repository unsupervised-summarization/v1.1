from typing import List, Dict
import torch

from .model import model, tokenizer, optimizer, scheduler
from .args import args
from .utils import set_device, to_numpy, Logger


class Trainer:
    def __init__(self):
        model.train()

        self.loss_logger = Logger('reconstructor-loss')
        self.lr_logger = Logger('reconstructor-lr')

        self.tokenizer = tokenizer
        self.model = set_device(model)

        # add special tokens
        self.add_tokens({
            'pad_token': '<|pad|>',
            'eos_token': '<|end|>',
            'sep_token': '<|sep|>'
        })
        model.resize_token_embeddings(len(self.tokenizer))

    def add_tokens(self, token_dict: Dict[str, str]):
        # Add special tokens if they were not defined.
        # ex) token_dict = {'pad_token': '<|pad|>'}
        for key, val in token_dict.items():
            if getattr(self.tokenizer, f'{key}_id') is None:
                self.tokenizer.add_special_tokens({key: val})

    @staticmethod
    def drop_long_texts(texts, len_func=None):
        # Drop texts longer than {args.max_length} tokens.
        if not args['drop_long_texts']:
            return None
        if len_func is None:
            # This expects that `texts` is zipped.
            def len_func(x):
                return len(x[0])
        return [tokens for tokens in texts if len_func(tokens) <= args['max_length']]

    def padding(self,
                input_ids: List[List[int]],
                token_type_ids: List[List[int]],
                labels: List[List[int]],
                max_len: int = None
                ) -> (List[List[int]], List[List[int]], List[List[int]], List[List[int]]):
        # padding

        # drop too long texts
        if args['drop_long_texts']:
            a = self.drop_long_texts(zip(input_ids, token_type_ids, labels))
            if len(a) == 0:
                return None
            input_ids, token_type_ids, labels = zip(*a)

        if max_len is None:
            max_len = max(map(len, input_ids))

        pad = self.tokenizer.pad_token_id
        attention_mask = [[1]*len(tokens) + [0]*(max_len-len(tokens)) for tokens in input_ids]
        input_ids = [tokens + [pad]*(max_len-len(tokens)) for tokens in input_ids]
        token_type_ids = [tokens + [pad]*(max_len-len(tokens)) for tokens in token_type_ids]
        labels = [tokens + [pad]*(max_len-len(tokens)) for tokens in labels]
        return attention_mask, input_ids, token_type_ids, labels

    def train_step(self,
                   summaries_tokens: List[List[int]],
                   documents_tokens: List[List[int]],
                   train: bool = True,
                   lr: float = None
                   ):
        # Tokens should be not padded.

        input_ids = [summary + [self.tokenizer.sep_token_id] + document
                     for summary, document in zip(summaries_tokens, documents_tokens)]
        labels = [[-100]*len(summary) + document + [self.tokenizer.eos_token_id]  # labels -100 will be masked (ignored).
                  for summary, document in zip(summaries_tokens, documents_tokens)]
        token_type_ids = [[0]*(len(summary)+1) + [1]*len(document)  # [0, 0, ..., 0, 1, 1, ..., 1]
                          for summary, document in zip(summaries_tokens, documents_tokens)]

        # padding
        result = self.padding(input_ids, token_type_ids, labels)
        if result is None:
            return 'No data'
        attention_mask, input_ids, token_type_ids, labels = result

        attention_mask = set_device(torch.tensor(attention_mask))
        input_ids = set_device(torch.tensor(input_ids))
        token_type_ids = set_device(torch.tensor(token_type_ids))
        labels = set_device(torch.tensor(labels))

        if lr is not None:
            # set learning rate of the optimizer
            for g in optimizer.param_groups:
                g['lr'] = lr

        # forward
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels, token_type_ids=token_type_ids)
        loss = outputs[0]

        # backward
        if train:
            loss.backward()
            optimizer.step()
            scheduler.step()

            # logging
            loss = float(to_numpy(loss))
            self.loss_logger(loss)
            self.lr_logger(float(scheduler.get_last_lr()[-1]))

        return loss

    def eval(self,
             summaries_tokens: List[List[int]],
             documents_tokens: List[List[int]]
             ):
        # Tokens should be not padded.

        input_ids = [summary + [self.tokenizer.sep_token_id] + document
                     for summary, document in zip(summaries_tokens, documents_tokens)]
        token_type_ids = [[0]*(len(summary)+1) + [1]*len(document)  # [0, 0, ..., 0, 1, 1, ..., 1]
                          for summary, document in zip(summaries_tokens, documents_tokens)]

        # padding
        attention_mask, input_ids, token_type_ids, _ = self.padding(input_ids, token_type_ids, input_ids)

        attention_mask = set_device(torch.tensor(attention_mask))
        input_ids = set_device(torch.tensor(input_ids))
        token_type_ids = set_device(torch.tensor(token_type_ids))

        # forward
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
