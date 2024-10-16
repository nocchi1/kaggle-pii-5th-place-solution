from omegaconf import DictConfig
from torch import nn
from torch.optim.optimizer import Optimizer

from src.train.optimizer import get_optimizer
from src.train.scheduler import get_scheduler


class ComponentFactory:
    # [TODO]要編集
    @staticmethod
    def get_model(config: DictConfig):
        if config.task_type == "detect":
            model = DetectModel(config)
        elif config.task_type == "classify":
            model = ClassifyModel()

        if config.reinit_layer_num > 0:
            model.reinit_layers(config.reinit_layer_num)
        if config.freeze_layer_num > 0:
            model.freeze_layers(config.freeze_layer_num)
        return model

    # [TODO]要編集
    @staticmethod
    def get_loss(config: DictConfig):
        if config.task_type == "detect":
            if config.smooth_type == "online":
                loss_fn = None
            else:
                loss_fn = SmoothingCELoss(config, weight=config.loss_class_weight)

        elif config.task_type == "classify":
            loss_fn = WeightedBCELoss()
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
    def get_scheduler(config: DictConfig, optimizer: Optimizer, steps_per_epoch: int):
        total_steps = config.epochs * steps_per_epoch
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
            first_cycle_steps = config.first_cycle_epochs * steps_per_epoch
            scheduler_args = {
                "first_cycle_steps": first_cycle_steps,
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
