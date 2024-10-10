import re
import statistics

import numpy as np

from src.utils.constant import TARGET2IDX_WITH_BIO, TARGET2IDX_WTO_BIO


def remove_target_prefix(data: list[dict]) -> list[dict]:
    for i in range(len(data)):
        labels = data[i]["labels"]
        data[i]["labels"] = [re.sub(r"(B-|I-)", "", label) for label in labels]
    return data


def convert_label_str2index(data: list[dict], remove_prefix: bool = False) -> list[dict]:
    if remove_prefix:
        data = remove_target_prefix(data)

    target2idx = TARGET2IDX_WTO_BIO if remove_prefix else TARGET2IDX_WITH_BIO
    for i in range(len(data)):
        labels = data[i]["labels"]
        data[i]["labels"] = [target2idx[label] for label in labels]
    return data


def mapping_index_org2char(
    full_texts: list[str],
    org_tokens: list[list[str]],
    org_tokens_idx: list[list[int]],
    whitespaces: list[list[bool]],
    fill_val: int = -1,
) -> list[np.ndarray]:
    """
    Map the index of the original token to each character position in the text.
    """
    char_indices = []
    for i in range(len(full_texts)):
        text = full_texts[i]
        org_token = org_tokens[i]
        org_token_idx = org_tokens_idx[i]
        whitespace = whitespaces[i]

        start_idx = 0
        char_idx = np.full(len(text), fill_val, dtype=np.int32)
        for i, t, s in zip(org_token_idx, org_token, whitespace):
            char_idx[start_idx : start_idx + len(t)] = i
            start_idx += len(t) + int(s)
        char_indices.append(char_idx)
    return char_indices


def mapping_index_org2char(
    full_texts: list[str],
    org_tokens: list[list[str]],
    org_tokens_idx: list[list[int]],
    whitespaces: list[list[bool]],
    fill_val: int = -1,
):
    """
    Map the index of the original token to each character position in the text.
    """
    char_indices = []
    for i in range(len(full_texts)):
        text = full_texts[i]
        org_token = org_tokens[i]
        org_token_idx = org_tokens_idx[i]
        whitespace = whitespaces[i]

        start_idx = 0
        char_idx = np.full(len(text), fill_val, dtype=np.int32)
        for i, t, s in zip(org_token_idx, org_token, whitespace):
            char_idx[start_idx : start_idx + len(t)] = i
            start_idx += len(t) + int(s)
        char_indices.append(char_idx)
    return char_indices


def mapping_index_char2token(
    char_indices: list[list[int]],
    input_ids: list[list[int]],
    offset_mappings: list[tuple[int, int]],
    fill_val: int = -1,
):
    """
    Map the index of the original token to the token position in the model, based on character position.
    """
    token_indices = []
    for i, (ids, maps) in enumerate(zip(input_ids, offset_mappings)):
        assert len(ids) == len(maps)
        char_idx = char_indices[i]
        token_idx = np.zeros(len(ids), dtype=np.int32)
        for idx, (start, end) in enumerate(maps):
            if start == 0 and end == 0:
                token_idx[idx] = fill_val
            else:
                map_idx = char_idx[start:end]
                token_idx[idx] = statistics.mode(map_idx)
        token_indices.append(token_idx)
    return token_indices


def mapping_index_char2token_overlapped(
    char_indices: list[list[int]],
    input_ids: list[list[int]],
    offset_mappings: list[tuple[int, int]],
    overlap_mapping: list[int],
    fill_val: int = -1,
):
    """
    Map the index of the original token to the token position in the model, based on character position.
    This function is used for the overlapped tokenization.
    """
    token_indices = []
    for idx, ids, maps in zip(overlap_mapping, input_ids, offset_mappings):
        assert len(ids) == len(maps)
        char_idx = char_indices[idx]
        token_idx = np.zeros(len(ids), dtype=np.int32)
        for i, (start, end) in enumerate(maps):
            if start == 0 and end == 0:
                token_idx[i] = fill_val
            else:
                map_idx = char_idx[start:end]
                token_idx[i] = statistics.mode(map_idx)
        token_indices.append(token_idx)
    return token_indices
