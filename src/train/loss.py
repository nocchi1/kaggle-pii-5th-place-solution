import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

TARGET_PAIR_DICT = {
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


class SmoothingCELoss(nn.Module):
    def __init__(self, config: DictConfig, class_weight: list[float] | None = None):
        super().__init__()
        if class_weight is not None:
            class_weight = torch.tensor(class_weight, dtype=torch.float, device=config.device)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weight)
        self.device = config.device

        self.class_num = config.class_num
        if config.remove_prefix and config.smooth_type == "weighted":
            config.smooth_type = "normal"
        self.smooth_type = config.smooth_type
        self.smooth_ratio = config.smooth_ratio
        self.smooth_pair = config.smooth_pair
        self.soft_matrix = self.get_soft_matrix()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred = y_pred.view(-1, y_pred.size(-1))
        y_true = y_true.view(-1)

        valid_idx = y_true != -1
        y_pred = y_pred[valid_idx]
        y_true = y_true[valid_idx]

        y_true = self.get_soft_label(y_true)
        return self.loss_fn(y_pred, y_true)

    def get_soft_label(self, y_true: torch.Tensor):
        if self.smooth_type in ["normal", "weighted"]:
            return self.soft_matrix[y_true]
        else:
            return y_true

    def get_soft_matrix(self):
        soft_matrix = torch.eye(self.class_num)

        if self.smooth_type == "normal":
            soft_matrix = soft_matrix * (1 - self.smooth_ratio) + self.smooth_ratio / self.class_num
            return soft_matrix.to(self.device)

        elif self.smooth_type == "weighted":
            for c, c_p in TARGET_PAIR_DICT.items():
                soft_label = soft_matrix[c]
                if c_p is not None:
                    soft_label[c_p] = self.smooth_pair

                soft_label = torch.where(soft_label == 0, self.smooth_ratio / self.class_num, soft_label)
                soft_label[c] = 1 - torch.sum(soft_label[soft_label != 1])
                soft_matrix[c] = soft_label
            return soft_matrix.to(self.device)
        else:
            return None


class OnlineSmoothingCELoss(nn.Module):
    def __init__(self, config: DictConfig, alpha: float = 0.50, class_weight: list[float] | None = None):
        super().__init__()
        class_weight = torch.tensor(class_weight, dtype=torch.float, device=config.device)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weight)
        self.device = config.device

        self.class_num = config.class_num
        self.alpha = alpha
        self.smooth_ratio = config.smooth_ratio

        self.stats_matrix = torch.zeros((config.class_num, config.class_num), device=config.device)
        self.counter = torch.zeros(config.class_num, device=config.device)

        # 最初は通常のLabel Smoothing
        self.soft_matrix = (
            torch.eye(config.class_num, device=config.device) * (1 - self.smooth_ratio)
            + self.smooth_ratio / config.class_num
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred = y_pred.view(-1, y_pred.size(-1))
        y_true = y_true.view(-1)
        self.accumulate_model_output(y_pred, y_true)

        valid_idx = y_true != -1
        y_pred = y_pred[valid_idx]
        y_true = y_true[valid_idx]

        y_true = self.get_soft_label(y_true)
        return self.loss_fn(y_pred, y_true)

    def get_soft_label(self, y_true: torch.Tensor):
        return self.soft_matrix[y_true]

    def accumulate_model_output(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred_class = torch.argmax(y_pred, dim=-1)
        for i in range(y_pred.size(0)):
            if y_pred_class[i] == y_true[i]:
                self.stats_matrix[y_true[i]] += F.softmax(y_pred[i], dim=-1)
                self.counter[y_true[i]] += 1

    def update_soft_matrix(self):
        soft_matrix = torch.zeros_like(self.stats_matrix)
        for i in range(self.class_num):
            if self.counter[i] > 0:
                hard_label = torch.eye(self.class_num, device=self.device)[i]
                soft_matrix[i] = hard_label * self.alpha + (self.stats_matrix[i] / self.counter[i]) * (1 - self.alpha)
            else:
                # soft_matrix[i] = torch.ones_like(soft_matrix[i]) / self.class_num # 論文の実装ではこちらが使われている
                soft_label = torch.eye(self.class_num, device=self.device)[i]
                soft_label = soft_label * (1 - self.smooth_ratio) + self.smooth_ratio / self.class_num
                soft_matrix[i] = soft_label

        self.soft_matrix = soft_matrix.detach()
        self.stats_matrix = torch.zeros((self.class_num, self.class_num)).to(self.device)
        self.counter = torch.zeros(self.class_num).to(self.device)
