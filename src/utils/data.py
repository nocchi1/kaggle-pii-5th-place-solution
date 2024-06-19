from typing import List
from src.utils.constant import TARGET2IDX_WITH_BIO, TARGET2IDX_WTO_BIO


def convert_label2index(data: List[dict], remove_prefix: bool = False) -> List[dict]:
    if remove_prefix:
        data = remove_target_prefix(data)
    target2idx = TARGET2IDX_WTO_BIO if remove_prefix else TARGET2IDX_WITH_BIO
    for i in range(len(data)):
        labels = data[i]['labels']
        data[i]['labels'] = [target2idx[label] for label in labels]
    return data

def remove_target_prefix(data: List[dict]) -> List[dict]:
    for i in range(len(data)):
        labels = data[i]['labels']
        data[i]['labels'] = [label.split('-')[-1] for label in labels]
    return data

