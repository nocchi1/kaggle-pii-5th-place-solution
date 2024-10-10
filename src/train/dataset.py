# ランダムな位置に出現するようにするギミック
# ラベルスムージングをどこで入れるか

import gc
import itertools
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

        # 外部データはidxがstr型なので全てstr型に変換されることに注意
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

        # 相対的な位置情報
        self.positions_ratio = [
            np.clip(np.array(org_idx) / org_tokens_len[str(doc_id)], 0, None)
            for org_idx, doc_id in zip(token_org_idx, self.overlap_doc_ids)
        ]
        # 絶対的な位置情報
        self.positions_abs = [
            np.clip(np.array(org_idx) / 3298, 0, None) for org_idx in token_org_idx
        ]  # 3298 is the maximum token length of dataset

        # ラベルを取得する
        if data_type in ["train", "valid"]:
            self.org_labels = [d["labels"] for d in data]
            org_labels_dict = {str(d["document"]): np.array(d["labels"]) for d in data}

            self.token_labels = []
            for org_idx, doc_id in zip(token_org_idx, self.overlap_doc_ids):
                label = org_labels_dict[str(doc_id)][org_idx]
                space_idx = np.where(np.array(org_idx) == -1)[0]
                label[space_idx] = -1  # -1になっている場合は-1を保持する必要がある
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
        追加学習を行う際に, 1段階目のみ使用するデータセットを削除する
        """
        assert self.data_type == "train"
        mask = self.overlap_additional
        self.input_ids = list(itertools.compress(self.input_ids, mask))
        self.attention_mask = list(itertools.compress(self.attention_mask, mask))
        self.positions_ratio = list(itertools.compress(self.positions_ratio, mask))
        self.positions_abs = list(itertools.compress(self.positions_abs, mask))
        self.token_labels = list(itertools.compress(self.token_labels, mask))
