from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.train.dataloader_utils import CollateFn, get_sampler, get_tokenizer
from src.train.dataset import ClassifyDataset, DetectDataset

__all__ = ["get_train_loaders", "get_full_train_loader"]


def get_train_loaders(config: DictConfig, data: list[dict]):
    tokenizer = get_tokenizer(config)
    has_position = True if config.task_type == "detect" else False
    collate_fn = CollateFn(tokenizer, is_train=True, has_position=has_position)

    dataloaders = []
    for fold in range(config.n_fold):
        if config.use_fold <= fold:
            break

        train_data, valid_data = [], []
        for d in data:
            if d["fold"] == fold:
                valid_data.append(d)
            else:
                train_data.append(d)

        if config.task_type == "detect":
            train_dataset = DetectDataset(config, train_data, tokenizer, data_type="train")
            valid_dataset = DetectDataset(config, valid_data, tokenizer, data_type="valid")
        elif config.task_type == "classify":
            train_dataset = ClassifyDataset(config, train_data, tokenizer, data_type="train")
            valid_dataset = ClassifyDataset(config, valid_data, tokenizer, data_type="valid")

        train_sampler = get_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train_batch,
            collate_fn=collate_fn,
            sampler=train_sampler,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.eval_batch,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        dataloaders.append((train_loader, valid_loader))
    return dataloaders


def get_full_train_loader(config, data: list[dict]) -> DataLoader:
    tokenizer = get_tokenizer(config)
    has_position = True if config.task_type == "detect" else False
    collate_fn = CollateFn(tokenizer, is_train=True, has_position=has_position)

    if config.task_type == "detect":
        train_dataset = DetectDataset(config, data, tokenizer, data_type="train")
    elif config.task_type == "classify":
        train_dataset = ClassifyDataset(config, data, tokenizer, data_type="train")

    train_sampler = get_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch,
        collate_fn=collate_fn,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader
