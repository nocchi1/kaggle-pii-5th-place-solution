import numpy as np
import torch
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler
from transformers import AutoTokenizer


def get_tokenizer(config: DictConfig) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if config.add_newline_token:
        tokenizer.add_tokens(["\n", "\r"], special_tokens=True)
    return tokenizer


def get_sampler(dataset: Dataset) -> Sampler:
    token_labels = dataset.token_labels
    is_pos_array = np.array([np.sum(np.where(label > 0, 1, 0)) > 0 for label in token_labels])
    pos_weight = len(is_pos_array) / np.sum(is_pos_array)
    weights = np.where(is_pos_array, pos_weight, 1.0)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)
    return sampler


class CollateFn:
    def __init__(self, tokenizer: AutoTokenizer, is_train: bool = True, has_position: bool = True):
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.has_position = has_position

    def __call__(self, batch):
        if self.has_position:
            if self.is_train:
                input_ids, attention_mask, positions_ratio, positions_abs, token_labels = zip(*batch)
            else:
                input_ids, attention_mask, positions_ratio, positions_abs = zip(*batch)
        elif self.is_train:
            input_ids, attention_mask, token_labels = zip(*batch)
        else:
            input_ids, attention_mask = zip(*batch)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        if self.is_train:
            token_labels = pad_sequence(token_labels, batch_first=True, padding_value=-1)

        if self.has_position:
            positions_ratio = pad_sequence(positions_ratio, batch_first=True, padding_value=0)
            positions_abs = pad_sequence(positions_abs, batch_first=True, padding_value=0)
            positions = torch.stack([positions_ratio, positions_abs], dim=-1)

        if self.has_position:
            if self.is_train:
                return input_ids, attention_mask, positions, token_labels
            else:
                return input_ids, attention_mask, positions
        elif self.is_train:
            return input_ids, attention_mask, token_labels
        else:
            return input_ids, attention_mask
