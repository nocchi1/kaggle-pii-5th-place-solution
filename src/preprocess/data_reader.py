import json
from pathlib import Path
from typing import Literal

from omegaconf import DictConfig

from src.preprocess.validation import split_validation
from src.utils.competition_utils import convert_label_str2index, load_json_data


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
        data = split_validation(data, n_splits=self.config.n_fold, seed=self.config.seed)
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
