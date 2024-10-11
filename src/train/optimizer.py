from torch import nn
from torch.optim import Adam, AdamW


def get_optimizer(
    model: nn.Module,
    optimizer_type: str,
    pretrained_lr: float,
    head_lr: float,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
):
    no_weight_decay = ["bias", "LayerNorm.weight"]
    parameters = []
    for name, params in model.named_parameters():
        if "backbone" in name:
            if not any([nd in name for nd in no_weight_decay]):
                parameters.append({"params": params, "weight_decay": weight_decay, "lr": pretrained_lr})
            else:
                parameters.append({"params": params, "weight_decay": 0.0, "lr": pretrained_lr})
        elif not any([nd in name for nd in no_weight_decay]):
            parameters.append({"params": params, "weight_decay": weight_decay, "lr": head_lr})
        else:
            parameters.append({"params": params, "weight_decay": 0.0, "lr": head_lr})

    if optimizer_type == "Adam":
        optimizer = Adam(parameters, lr=head_lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_type == "AdamW":
        optimizer = AdamW(parameters, lr=head_lr, weight_decay=weight_decay, betas=betas)
    return optimizer
