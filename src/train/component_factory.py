from omegaconf import DictConfig
from torch.optim.optimizer import Optimizer

from src.model.model import ClassifyModel, DetectModel
from src.train.loss import MaskedBCELoss, OnlineSmoothingCELoss, SmoothingCELoss
from src.train.optimizer import get_optimizer
from src.train.scheduler import get_scheduler

__all__ = ["ComponentFactory"]


class ComponentFactory:
    @staticmethod
    def get_model(config: DictConfig):
        if config.task_type == "detect":
            model = DetectModel(config)
        elif config.task_type == "classify":
            model = ClassifyModel(config)

        if config.reinit_layer_num > 0:
            model.reinit_layers(config.reinit_layer_num)
        if config.freeze_layer_num > 0:
            model.freeze_layers(config.freeze_layer_num)
        return model

    @staticmethod
    def get_loss(config: DictConfig):
        if config.task_type == "detect":
            class_weight = [1] + [config.positive_class_weight] * (config.class_num - 1)
            if config.smooth_type == "online":
                loss_fn = OnlineSmoothingCELoss(
                    config,
                    alpha=0.5,  # Can be further tuned
                    class_weight=class_weight,
                )
            else:
                loss_fn = SmoothingCELoss(config, class_weight=class_weight)
        elif config.task_type == "classify":
            loss_fn = MaskedBCELoss(config)
        return loss_fn

    @staticmethod
    def get_optimizer(config: DictConfig, model):
        optimizer = get_optimizer(
            model,
            optimizer_type=config.optimizer_type,
            pretrained_lr=config.pretrained_lr,
            head_lr=config.head_lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )
        return optimizer

    @staticmethod
    def get_scheduler(config: DictConfig, optimizer: Optimizer, total_steps: int):
        if config.scheduler_type == "linear":
            scheduler_args = {
                "num_warmup_steps": config.num_warmup_steps,
                "num_training_steps": total_steps,
            }
        elif config.scheduler_type == "cosine":
            scheduler_args = {
                "num_warmup_steps": config.num_warmup_steps,
                "num_training_steps": total_steps,
                "num_cycles": config.num_cycles,
            }
        elif config.scheduler_type == "cosine_custom":
            scheduler_args = {
                "first_cycle_steps": total_steps,
                "cycle_factor": config.cycle_factor,
                "num_warmup_steps": config.num_warmup_steps,
                "min_lr": config.min_lr,
                "gamma": config.gamma,
            }
        elif config.scheduler_type == "reduce_on_plateau":
            scheduler_args = {
                "mode": config.mode,
                "factor": config.factor,
                "patience": config.patience,
                "min_lr": config.min_lr,
            }
        else:
            raise ValueError(f"Invalid scheduler_type: {config.scheduler_type}")

        scheduler = get_scheduler(optimizer, scheduler_type=config.scheduler_type, scheduler_args=scheduler_args)
        return scheduler
