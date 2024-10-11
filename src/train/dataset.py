import gc
import itertools
import random
from typing import Literal

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.utils.competition_utils import (
    mapping_index_char2token,
    mapping_index_char2token_overlapped,
    mapping_index_org2char,
)


class DetectDataset(Dataset):
    def __init__(
        self,
        config: DictConfig,
        data: list[dict],
        tokenizer: AutoTokenizer,
        data_type: Literal["train", "valid", "test"],
    ):
        self.config = config
        self.data_type = data_type
        self.tokenizer = tokenizer
        self.doc_ids = [d["document"] for d in data]
        self.full_texts = [d["full_text"] for d in data]
        self.org_tokens = [d["tokens"] for d in data]
        self.whitespaces = [d["trailing_whitespace"] for d in data]
        self.additionals = [d["additional"] for d in data]

        stride = config.train_stride if data_type == "train" else config.eval_stride
        tokens = tokenizer(
            self.full_texts,
            max_length=config.max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            truncation=True,
        )
        self.input_ids = tokens["input_ids"]
        self.attention_mask = tokens["attention_mask"]
        self.offset_mapping = tokens["offset_mapping"]
        self.overflow_mapping = tokens["overflow_to_sample_mapping"]

        # Note that external data has an idx of type str, so all idxs are converted to type str
        self.overlap_doc_ids = np.array(self.doc_ids)[self.overflow_mapping]
        self.overlap_additional = np.array(self.additionals)[self.overflow_mapping]

        org_tokens_idx = [list(range(len(d["tokens"]))) for d in data]
        org_tokens_len = {str(d["document"]): len(d["tokens"]) for d in data}

        char_org_idx = mapping_index_org2char(
            self.full_texts,
            self.org_tokens,
            org_tokens_idx,
            self.whitespaces,
            fill_val=-1,
        )
        token_org_idx = mapping_index_char2token_overlapped(
            char_org_idx,
            self.input_ids,
            self.offset_mapping,
            self.overflow_mapping,
            fill_val=-1,
        )

        self.positions_ratio = [
            np.clip(np.array(org_idx) / org_tokens_len[str(doc_id)], 0, None)
            for org_idx, doc_id in zip(token_org_idx, self.overlap_doc_ids)
        ]
        self.positions_abs = [
            np.clip(np.array(org_idx) / 3298, 0, None) for org_idx in token_org_idx
        ]  # 3298 is the maximum token length of dataset

        if data_type in ["train", "valid"]:
            self.org_labels = [d["labels"] for d in data]
            org_labels_dict = {str(d["document"]): np.array(d["labels"]) for d in data}

            self.token_labels = []
            for org_idx, doc_id in zip(token_org_idx, self.overlap_doc_ids):
                label = org_labels_dict[str(doc_id)][org_idx]
                space_idx = np.where(np.array(org_idx) == -1)[0]
                label[space_idx] = -1  # In the case of -1, we need to keep it as -1
                self.token_labels.append(label)

    def __getitem__(self, idx: int):
        if self.data_type in ["train", "valid"]:
            return (
                torch.tensor(self.input_ids[idx], dtype=torch.long, device="cpu"),
                torch.tensor(self.attention_mask[idx], dtype=torch.long, device="cpu"),
                torch.tensor(self.positions_ratio[idx], dtype=torch.float, device="cpu"),
                torch.tensor(self.positions_abs[idx], dtype=torch.float, device="cpu"),
                torch.tensor(self.token_labels[idx], dtype=torch.long, device="cpu"),
            )
        else:
            return (
                torch.tensor(self.input_ids[idx], dtype=torch.long, device="cpu"),
                torch.tensor(self.attention_mask[idx], dtype=torch.long, device="cpu"),
                torch.tensor(self.positions_ratio[idx], dtype=torch.float, device="cpu"),
                torch.tensor(self.positions_abs[idx], dtype=torch.float, device="cpu"),
            )

    def __len__(self):
        return len(self.input_ids)

    def drop_first_only_data(self):
        """
        Remove the dataset used only in the first phase when performing additional training
        """
        assert self.data_type == "train"
        mask = self.overlap_additional
        self.input_ids = list(itertools.compress(self.input_ids, mask))
        self.attention_mask = list(itertools.compress(self.attention_mask, mask))
        self.positions_ratio = list(itertools.compress(self.positions_ratio, mask))
        self.positions_abs = list(itertools.compress(self.positions_abs, mask))
        self.token_labels = list(itertools.compress(self.token_labels, mask))


# Accuracy did not improve, so this is not used
class DetectRandomDataset(Dataset):
    """
    Dataset that performs random sampling
    For positive cases: Randomly sample, ensuring the start position of PII is included
    For negative cases: Randomly sample around a random position from random text
    """

    def __init__(
        self,
        config: DictConfig,
        data: list[dict],
        tokenizer: AutoTokenizer,
        data_type: Literal["train", "valid", "test"] = "train",
        positive_ratio: float = 0.50,
    ):
        assert data_type == "train"  # This dataset is used only for training

        self.config = config
        self.positive_ratio = positive_ratio
        self.data_type = data_type
        self.tokenizer = tokenizer
        self.doc_ids = [d["document"] for d in data]
        self.full_texts = [d["full_text"] for d in data]
        self.org_tokens = [d["tokens"] for d in data]
        self.whitespaces = [d["trailing_whitespace"] for d in data]
        self.additionals = [d["additional"] for d in data]

        tokens = tokenizer(self.full_texts, return_offsets_mapping=True)
        self.input_ids = tokens["input_ids"]
        self.attention_mask = tokens["attention_mask"]
        self.offset_mapping = tokens["offset_mapping"]

        org_tokens_idx = [list(range(len(d["tokens"]))) for d in data]
        org_tokens_len = {str(d["document"]): len(d["tokens"]) for d in data}

        char_org_idx = mapping_index_org2char(
            self.full_texts,
            self.org_tokens,
            org_tokens_idx,
            self.whitespaces,
            fill_val=-1,
        )
        token_org_idx = mapping_index_char2token(
            char_org_idx,
            self.input_ids,
            self.offset_mapping,
            fill_val=-1,
        )

        self.positions_rel = [
            np.clip(np.array(org_idx) / org_tokens_len[str(doc_id)], 0, None)
            for org_idx, doc_id in zip(token_org_idx, self.doc_ids)
        ]
        self.positions_abs = [
            np.clip(np.array(org_idx) / 3298, 0, None) for org_idx in token_org_idx
        ]  # 3298 is the maximum token length of dataset

        if self.data_type in ["train", "valid"]:
            self.org_labels = [d["labels"] for d in data]
            org_labels_dict = {str(d["document"]): np.array(d["labels"]) for d in data}

            self.token_labels = []
            for org_idx, doc_id in zip(token_org_idx, self.doc_ids):
                label = org_labels_dict[doc_id][org_idx]
                space_idx = np.where(np.array(org_idx) == -1)[0]
                label[space_idx] = -1  # In the case of -1, we need to keep it as -1
                self.token_labels.append(label)

            self.pii_locs = self.find_pii_locations(self.doc_ids, self.token_labels)

    def __getitem__(self, idx: int):
        if random.random() >= self.positive_ratio:
            global_idx, _, pii_idx = self.pii_locs[idx]
        else:
            global_idx = random.choice(range(len(self.input_ids)))
            pii_idx = random.choice(range(len(self.input_ids[global_idx])))

        input_ids = self.input_ids[global_idx]
        attention_mask = self.attention_mask[global_idx]
        positions_rel = self.positions_rel[global_idx]
        positions_abs = self.positions_abs[global_idx]
        token_labels = self.token_labels[global_idx]

        left_length = random.randint(0, self.config.max_length - 1)
        start_idx = max(0, pii_idx - left_length)
        end_idx = start_idx + self.config.max_length
        ex_right_length = max(0, end_idx - len(input_ids))
        start_idx = max(0, start_idx - ex_right_length)
        end_idx = end_idx - ex_right_length

        assert start_idx >= 0 and end_idx <= len(input_ids)
        assert start_idx <= pii_idx and pii_idx <= end_idx

        input_ids = input_ids[start_idx:end_idx]
        attention_mask = attention_mask[start_idx:end_idx]
        positions_rel = positions_rel[start_idx:end_idx]
        positions_abs = positions_abs[start_idx:end_idx]
        token_labels = token_labels[start_idx:end_idx]

        return (
            torch.tensor(input_ids, dtype=torch.long, device="cpu"),
            torch.tensor(attention_mask, dtype=torch.long, device="cpu"),
            torch.tensor(positions_rel, dtype=torch.float, device="cpu"),
            torch.tensor(positions_abs, dtype=torch.float, device="cpu"),
            torch.tensor(token_labels, dtype=torch.long, device="cpu"),
        )

    def __len__(self):
        return len(self.pii_locs)

    def find_pii_locations(self, doc_ids: list[int], token_labels: list[int]) -> list[list[int]]:
        """
        Identify the index of PII starting positions for each document
        """
        pii_locs = []
        for i, (ids, labels) in enumerate(zip(doc_ids, token_labels)):
            is_pii_array = np.where(labels != 0, 1, 0)
            is_pii_diff = np.where(np.diff(is_pii_array, prepend=0) == 1, 1, 0)
            pii_index = np.where(is_pii_diff == 1)[0]
            for p_idx in pii_index:
                pii_locs.append([i, ids, p_idx])
        return pii_locs
