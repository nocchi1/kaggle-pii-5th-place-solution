import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def split_validation(data: list[dict], n_splits: int, seed: int = 10) -> np.ndarray:
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
