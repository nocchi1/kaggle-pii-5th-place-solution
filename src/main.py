import argparse
import gc
import pickle
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.postprocess import PostProcessor
from src.preprocess import DetectDataReader
from src.train import Trainer, get_full_train_loader, get_train_loaders
from src.train.dataloader_utils import CollateFn, get_sampler, get_tokenizer
from src.utils import TimeUtil, get_config, get_logger, seed_everything
from src.utils.competition_utils import get_pred_df, get_truth_df
from src.utils.metric import evaluate_metric, get_best_negative_threshold


def main():
    # Setup
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--config_name", help="config file name", type=str, required=True)
    parser.add_argument("-d", "--debug", help="debug mode", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    config = get_config(args.config_name, config_dir=Path("./config"))
    config.debug = args.debug
    logger = get_logger(config.output_path)
    logger.info(f"EXP:{config.exp} Start...")
    seed_everything(config.seed)

    # Overwrite Config for Debug
    if config.debug:
        config.eval_steps = 100
        config.ema_update_after_step = 100
        config.epochs = 2
        config.add_epochs = 2

    # Load Data
    reader = DetectDataReader(config, "train")
    data = reader.load_data()
    logger.info(f"Data Size: {len(data)}")

    # Create Dataloader
    dataloaders = get_train_loaders(config, data)

    # K-Fold Training -> ベストなステップ数とOOFに対する予測値を取ることが目的
    oof_dfs = []
    best_steps, best_add_steps = [], []
    collate_fn = CollateFn(get_tokenizer(config), is_train=True)

    for fold, (train_loader, valid_loader) in enumerate(dataloaders):
        logger.info(f"FOLD{fold} : Training Start...")

        # First Training
        trainer = Trainer(config, logger, save_suffix=f"_fold{fold}")
        best_score, best_steps_, oof_df = trainer.train(train_loader, valid_loader)
        if config.smooth_type == "online":
            loss_soft_matrix = trainer.loss_fn.soft_matrix.clone()
        best_steps.append(best_steps_)
        logger.info(f"FOLD{fold} : First Training Done! -->> Best Score: {best_score}, Best Steps: {best_steps_}")

        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        if config.add_train:
            # Create High-Quality Dataloader
            train_dataset = train_loader.dataset
            train_dataset.drop_first_only_data()
            train_loader = DataLoader(
                train_dataset,
                sampler=get_sampler(train_dataset),
                batch_size=config.train_batch,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=True,
            )

            # Additional Training
            trainer = Trainer(config, logger, save_suffix=f"_fold{fold}")
            if config.smooth_type == "online":
                trainer.loss_fn.soft_matrix = loss_soft_matrix
            best_score, best_add_steps_, add_oof_df = trainer.train(
                train_loader,
                valid_loader,
                retrain=True,
                retrain_weight_name=f"model_fold{fold}_best",
                retrain_best_score=best_score,
            )
            best_add_steps.append(best_add_steps_)
            logger.info(
                f"FOLD{fold} : Additional Training Done! -->> Best Score: {best_score}, Best Add Steps: {best_add_steps_}"
            )
            if add_oof_df is not None:
                oof_df = add_oof_df.clone()

            del trainer, train_dataset, add_oof_df
            gc.collect()
            torch.cuda.empty_cache()

        oof_df.write_parquet(config.output_path / f"oof_fold{fold}.parquet")
        oof_dfs.append(oof_df)

        del train_loader, valid_loader, oof_df
        gc.collect()
        torch.cuda.empty_cache()

    del dataloaders
    gc.collect()

    # Save OOF
    oof_df = pl.concat(oof_dfs)
    oof_df.write_parquet(config.output_path / "oof.parquet")
    del oof_dfs
    gc.collect()
    oof_fold_files = config.output_path.glob("oof_fold*.parquet")
    for f in oof_fold_files:
        Path(f).unlink()

    # Get Best Negative Threshold
    truth_df = get_truth_df(config, oof_df["document"].unique().to_list(), convert_idx=False)
    best_score, best_th = get_best_negative_threshold(config, oof_df, truth_df)
    message = f"Overall OOF Best Score: {best_score}, Best Negative Threshold: {best_th}"
    logger.info(message)
    config.negative_th = best_th.item()

    # Full Training
    if config.full_train:
        logger.info("Full Train : Training Start...")
        train_loader = get_full_train_loader(config, data)

        # First Training
        full_steps = np.max(best_steps)
        trainer = Trainer(config, logger, save_suffix="")
        trainer.train(train_loader, valid_loader=None, full_train=True, full_steps=full_steps)
        if config.smooth_type == "online":
            loss_soft_matrix = trainer.loss_fn.soft_matrix.clone()
        logger.info("Full Train : First Training Done!")

        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        if config.add_train:
            full_add_steps = np.max(best_add_steps)
            if full_add_steps > 0:
                # Create High-Quality Dataloader
                train_dataset = train_loader.dataset
                train_dataset.drop_first_only_data()
                train_loader = DataLoader(
                    train_dataset,
                    sampler=get_sampler(train_dataset),
                    batch_size=config.train_batch,
                    collate_fn=collate_fn,
                    pin_memory=True,
                    drop_last=True,
                )

                # Additional Training
                trainer = Trainer(config, logger, save_suffix="")
                if config.smooth_type == "online":
                    trainer.loss_fn.soft_matrix = loss_soft_matrix
                trainer.train(
                    train_loader,
                    valid_loader=None,
                    retrain=True,
                    retrain_weight_name="model_full",
                    full_train=True,
                    full_steps=full_add_steps,
                )
                logger.info("Full Train : Additional Training Done!")
                del trainer, train_dataset
                gc.collect()
                torch.cuda.empty_cache()

        del train_loader
        gc.collect()
        torch.cuda.empty_cache()

    # Post-Process
    pred_df = get_pred_df(oof_df, config.class_num, negative_th=config.negative_th)
    truth_df = get_truth_df(config, pred_df["document"].unique().to_list(), convert_idx=True)

    pper = PostProcessor(config)
    pred_df = pper.post_process(pred_df)
    score = evaluate_metric(pred_df, truth_df)
    logger.info(f"OOF Score after Post-Process: {score:.5f}")


if __name__ == "__main__":
    main()
