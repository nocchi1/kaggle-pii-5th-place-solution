import re
from typing import List

from src.utils.constant import TARGET2IDX_WITH_BIO, TARGET2IDX_WTO_BIO


def remove_target_prefix(data: list[dict]) -> list[dict]:
    for i in range(len(data)):
        labels = data[i]["labels"]
        data[i]["labels"] = [re.sub(r"(B-|I-)", "", label) for label in labels]
    return data


def convert_label2index(data: list[dict], remove_prefix: bool = False) -> list[dict]:
    if remove_prefix:
        data = remove_target_prefix(data)

    target2idx = TARGET2IDX_WTO_BIO if remove_prefix else TARGET2IDX_WITH_BIO
    for i in range(len(data)):
        labels = data[i]["labels"]
        data[i]["labels"] = [target2idx[label] for label in labels]
    return data
