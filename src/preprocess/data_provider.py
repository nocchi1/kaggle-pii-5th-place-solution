import json
from pathlib import PosixPath
from typing import List, Literal

from omegaconf import DictConfig

from src.preprocess.validation import split_validation
from src.utils.competition_utils import convert_label_str2index


class DataProvider1st:
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
        data = self.load_json_data(self.config.input_path / "train.json")
        data = convert_label_str2index(data)
        data = split_validation(data, n_splits=self.config.n_fold, seed=self.config.seed)
        for d in data:
            d["additional"] = 1

        for name, add_flag in zip(self.exter_names, self.exter_add_flags):
            file_path = self.config.input_path / "external" / f"{name}.json"
            data_ = self.load_json_data(file_path)
            data_ = convert_label_str2index(data_)
            additional = 1 if add_flag else 0
            for d in data_:
                d["additional"] = additional
                d["fold"] = -1
            data.extend(data_)
        return data

    def load_test_data(self):
        data = self.load_json_data(self.config.input_path / "train.json")
        return data

    def load_json_data(self, file_path: PosixPath) -> list[dict]:
        with open(file_path) as f:
            data = json.load(f)
        if self.config.debug:
            data = data[:100]
        return data
