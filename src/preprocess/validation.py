from collections import Counter

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold

__all__ = ["detect_split_validation", "classify_split_validation"]


def detect_split_validation(data: list[dict], n_splits: int, seed: int = 10) -> np.ndarray:
    target_max = max(list(map(lambda x: max(x["labels"]), data)))
    target_table = np.zeros((len(data), target_max + 1))

    for i, d in enumerate(data):
        target_unq = np.unique(d["labels"])
        target_table[i, target_unq] = 1
    target_table = target_table[:, 1:]

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.ones(len(data)) * -1
    for fold, (_, valid_idx) in enumerate(mskf.split(target_table, target_table)):
        folds[valid_idx] = fold

    for d, fold in zip(data, folds):
        d["fold"] = fold
    return data


def classify_split_validation(data: list[dict], n_splits: int, seed: int = 10) -> np.ndarray:
    y = []
    for d in data:
        label_counter = Counter(d["second_labels"])
        if 0 in label_counter and 1 in label_counter:
            y.append(0)
        elif 0 in label_counter:
            y.append(1)
        elif 1 in label_counter:
            y.append(2)
        else:
            y.append(3)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.ones(len(data)) * -1
    for fold, (_, valid_idx) in enumerate(skf.split(data, y=y)):
        folds[valid_idx] = fold

    for d, fold in zip(data, folds):
        d["fold"] = fold
    return data
