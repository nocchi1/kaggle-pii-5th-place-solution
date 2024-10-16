import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

# [TODO]OnelineSmoothingCELossクラスを実装してください
# [TODO]2ndStageのLoss関数を実装してください


class SmoothingCELoss(nn.Module):
    def __init__(self, config: DictConfig, weight: list[float]):
        super().__init__()
        self.device = config.device
        weight = torch.tensor(weight, dtype=torch.float, device=config.device)
        self.loss = nn.CrossEntropyLoss(weight, ignore_index=-1)

        self.class_num = config.class_num
        if config.remove_prefix and config.smooth_type == "weighted":
            config.smooth_type = "normal"
        self.smooth_type = config.smooth_type
        self.smooth_ratio = config.smooth_ratio
        self.smooth_pair = config.smooth_pair
        self.target_pair_dict = {
            0: None,
            1: 8,
            2: None,
            3: None,
            4: 9,
            5: 10,
            6: 11,
            7: 12,
            8: 1,
            9: 4,
            10: 5,
            11: 6,
            12: 7,
        }
        self.target_matrix = self.get_target_matrix()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred = y_pred.view(-1, y_pred.size(-1))
        y_true = y_true.view(-1)
        y_true = self.get_smooth_label(y_true)
        return self.loss(y_pred, y_true)

    def get_smooth_label(self, y_true: torch.Tensor):
        if self.smooth_type in ["normal", "weighted"]:
            return self.target_matrix[y_true]
        else:
            return y_true

    def get_target_matrix(self):
        target_matrix = torch.eye(self.class_num)
        for c, c_p in self.target_pair_dict.items():
            target_vector = target_matrix[c]
            if c_p is not None:
                if self.smooth_type == "weighted":
                    target_vector[c_p] = self.smooth_pair

            target_vector = torch.where(target_vector == 0, self.smooth_ratio / self.class_num, target_vector)
            target_vector[c] = 1 - torch.sum(target_vector[target_vector != 1])
            target_matrix[c] = target_vector
        return target_matrix.to(self.device)
