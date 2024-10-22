from typing import Literal

import polars as pl
from omegaconf import DictConfig

from src.preprocess.validation import classify_split_validation, detect_split_validation
from src.utils.competition_utils import (
    convert_label_str2index,
    get_first_pred_df,
    load_json_data,
)

__all__ = ["DetectDataReader", "ClassifyDataReader"]


class DetectDataReader:
    def __init__(self, config: DictConfig, data_type: Literal["train", "test"]):
        self.config = config
        self.data_type = data_type
        if data_type == "train":
            self.exter_names = [d[0] for d in config.exter_dataset]
            self.exter_add_flags = [d[1] for d in config.exter_dataset]

    def load_data(self):
        if self.data_type == "train":
            data = self.load_train_data()
        elif self.data_type == "test":
            data = self.load_test_data()
        return data

    def load_train_data(self):
        data = load_json_data(self.config.input_path / "train.json", debug=self.config.debug)
        data = convert_label_str2index(data, self.config.remove_prefix)
        data = detect_split_validation(data, n_splits=self.config.n_fold, seed=self.config.seed)
        for d in data:
            d["additional"] = 1

        for name, add_flag in zip(self.exter_names, self.exter_add_flags):
            file_path = self.config.input_path / "external" / f"{name}.json"
            data_ = load_json_data(file_path, debug=self.config.debug)
            data_ = convert_label_str2index(data_, self.config.remove_prefix)
            additional = 1 if add_flag else 0
            for d in data_:
                d["additional"] = additional
                d["fold"] = -1
            data.extend(data_)
        return data

    def load_test_data(self):
        data = load_json_data(self.config.input_path / "test.json")
        return data


class ClassifyDataReader:
    def __init__(self, config: DictConfig, data_type: Literal["train", "test"]):
        self.config = config
        self.data_type = data_type

    def load_data(self, first_exp: str):
        data = load_json_data(file_path=self.config.input_path / f"{self.data_type}.json", debug=self.config.debug)
        doc_ids = [d["document"] for d in data]
        if self.data_type == "train":
            data = convert_label_str2index(data, remove_prefix=False)

        pred_df = get_first_pred_df(
            self.config,
            oof_file_path=self.config.output_path.parent / first_exp / "oof.parquet",
            negative_th=self.config.first_negative_th,
        )
        pred_df = pred_df.sort("document", "token_idx")
        pred_df = pred_df.with_columns(name_pred=pl.col("pred").map_elements(lambda x: 1 if x in [1, 8] else 0))

        data_ = []
        for (pred_id,), pred_df_ in pred_df.group_by("document"):
            idx = doc_ids.index(pred_id)
            d = data[idx]
            assert len(d["tokens"]) == len(pred_df_)

            d["name_pred"] = pred_df_["name_pred"].to_list()  # TP + FP

            if self.data_type == "train":
                second_labels, name_pred = [], []
                for label, pred in zip(d["labels"], d["name_pred"]):
                    if label in [1, 8]:  # TP + FN
                        second_labels.append(1)
                        name_pred.append(1)
                    elif label not in [1, 8] and pred in [1, 8]:  # FP
                        second_labels.append(0)
                        name_pred.append(1)
                    else:  # TN
                        second_labels.append(-1)
                        name_pred.append(0)

                d["second_labels"] = second_labels
                d["name_pred"] = name_pred

            data_.append(d)

        if self.data_type == "train":
            data = classify_split_validation(data_, n_splits=self.config.n_fold, seed=self.config.seed)
        return data
