import math

import torch
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


class CustomCosineAnnealing(LRScheduler):
    """
    Custom implementation of cosine annealing.
    Allows for the decay of the maximum learning rate in cycles after the second cycle.

    This implementation is a modification of the following:
    ref: https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup

    Fixes:
        - Fixed a bug where max_lr was unified internally when lr was specified for each layer
        - Fixed a bug where the learning rate would drop below min_lr
        - Fixed a bug that occurred when last_epoch was specified

    args:
        optimizer: torch.optim.Optimizer
        first_cycle_steps (int): Number of steps in the first cosine cycle
        cycle_factor (float): Factor by How the cycle length is extended in subsequent cycles
        num_warmup_steps (int): Number of steps used for warmup
        min_lr (float): Minimum learning rate
        gamma (float): Decay rate of the maximum learning rate
        last_epoch (int): The step number from which to resume training
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_factor: float = 1.0,
        num_warmup_steps: int = 0,
        min_lr: float = 0.0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        if first_cycle_steps <= num_warmup_steps:
            num_warmup_steps = first_cycle_steps // 2

        self.optimizer = optimizer
        self.first_cycle_steps = first_cycle_steps
        self.cycle_factor = cycle_factor
        self.num_warmup_steps = num_warmup_steps
        self.min_lr = min_lr
        self.gamma = gamma

        self.cycle = 0
        self.current_cycle_steps = first_cycle_steps
        self.steps_in_cycle = last_epoch

        self.max_lrs = []
        for param_group in self.optimizer.param_groups:
            self.max_lrs.append(param_group["lr"])
        self.current_max_lrs = self.max_lrs.copy()

        self.init_lr()
        super().__init__(optimizer, last_epoch)

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            param_group["initial_lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.steps_in_cycle == -1:
            return self.base_lrs
        elif self.steps_in_cycle < self.num_warmup_steps:
            return [
                (max_lr - base_lr) * self.steps_in_cycle / self.num_warmup_steps + base_lr
                for base_lr, max_lr in zip(self.base_lrs, self.current_max_lrs)
            ]
        else:
            return [
                (
                    base_lr
                    + (max_lr - base_lr)
                    * (
                        1
                        + math.cos(
                            math.pi
                            * (self.steps_in_cycle - self.num_warmup_steps)
                            / (self.current_cycle_steps - self.num_warmup_steps)
                        )
                    )
                    / 2
                )
                for base_lr, max_lr in zip(self.base_lrs, self.current_max_lrs)
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        if epoch >= self.first_cycle_steps:
            if self.cycle_factor == 1.0:
                self.steps_in_cycle = epoch % self.first_cycle_steps
                self.cycle = epoch // self.first_cycle_steps
            else:
                n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_factor - 1) + 1), self.cycle_factor))
                self.cycle = n
                self.steps_in_cycle = epoch - int(
                    self.first_cycle_steps * (self.cycle_factor**n - 1) / (self.cycle_factor - 1)
                )
                self.current_cycle_steps = self.first_cycle_steps * self.cycle_factor ** (n)
        else:
            self.current_cycle_steps = self.first_cycle_steps
            self.steps_in_cycle = epoch

        self.current_max_lrs = [max(self.min_lr, lr * (self.gamma**self.cycle)) for lr in self.max_lrs]
        self.last_epoch = epoch

        current_lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, current_lrs):
            param_group["lr"] = lr


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    scheduler_args: dict,
):
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, **scheduler_args)
    elif scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, **scheduler_args)
    elif scheduler_type == "cosine_custom":
        scheduler = CustomCosineAnnealing(optimizer, **scheduler_args)
    elif scheduler_type == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(optimizer, **scheduler_args)
    else:
        raise ValueError(f"Invalid scheduler type: {scheduler_type}")
    return scheduler
