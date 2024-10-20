#!/usr/bin/env python

# # Library

# In[3]:


# %%writefile ../config/exp_087.yaml
# exp: "087"
# debug: false
# seed: 10
# task_type: "detect"
# device: "cuda"

# # data preprocess
# remove_prefix: true
# exter_dataset:
#   - ["nicholas", true]
#   - ["mpware", false]
#   - ["pjma", false]

# n_fold: 3
# use_fold: 3

# # dataset, dataloader
# add_newline_token: true
# max_length: 128
# train_stride: 96
# eval_stride: 64
# train_batch: 16
# eval_batch: 64

# # model
# model_path: "microsoft/deberta-v3-large"
# class_num: 8 # with prefix -> 13, without prefix -> 8
# lstm_type: "none"
# use_hidden_states: 2
# dropout: 0.10
# hidden_dropout: 0.10
# attention_dropout: 0.10
# reinit_layer_num: 0
# freeze_layer_num: 0

# # loss
# smooth_type: "online"
# smooth_ratio: 0.05
# smooth_pair: 0.05
# positive_class_weight: 10

# # optimizer
# optimizer_type: "AdamW"
# pretrained_lr: 1e-6
# head_lr: 1e-4
# weight_decay: 0.01
# betas: [0.9, 0.999]

# # scheduler
# scheduler_type: "cosine_custom"
# first_cycle_epochs: 4
# cycle_factor: 1
# num_warmup_steps: 0
# min_lr: 1e-9
# gamma: 1.0

# # training
# epochs: 4
# accumulation_steps: 2
# eval_steps: 1000
# negative_th: 0.660
# negative_th_method: "overall"
# amp: true
# ema: false
# ema_decay: 0.999
# ema_update_after_step: 8000

# # additional training
# add_train: true
# add_epochs: 4
# add_first_cycle_epochs: 4

# # full training
# full_train: true


# In[1]:


import gc
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

sys.path.append("..")

from src.preprocess import DetectDataProvider
from src.train import get_full_train_loader, get_train_loaders
from src.train.dataloader_utils import CollateFn, get_sampler, get_tokenizer
from src.utils import TimeUtil, get_config, get_logger, seed_everything

# # Setup

# In[2]:


# [TODO]コマンドライン引数
exp = "087"
debug = False


# In[4]:


config = get_config(exp, config_dir=Path("../config"))
logger = get_logger(config.output_path)
logger.info(f"exp:{exp} start")

seed_everything(config.seed)


# In[5]:


# [TODO]
config.debug = debug
config.use_fold = 3
config.eval_steps = 500  # 100
config.ema_update_after_step = 100

config.epochs = 2
config.first_cycle_epochs = 2
config.add_epochs = 2
config.add_first_cycle_epochs = 2


# # Data

# In[6]:


dpr = DetectDataProvider(config, "train")
data = dpr.load_data()
logger.info(f"Data Size: {len(data)}")


# In[7]:


# [TODO]データサイズを調整する

data_ = []
for fold in [-1, 0, 1, 2]:
    fold_data = [d for d in data if d["fold"] == fold]
    fold_data = fold_data[:100]
    data_.extend(fold_data)

data = data_
len(data)


# In[8]:


# dataloaders = get_train_loaders(config, data)


# # Model

# In[9]:


import loguru
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader

from src.train.component_factory import ComponentFactory
from src.train.ema import ModelEmaV3
from src.utils.competition_utils import (
    get_char2org_df,
    get_char_pred_df,
    get_original_token_df,
    get_pred_df,
    get_truth_df,
    restore_prefix,
)
from src.utils.metric import evaluate_metric, get_best_negative_threshold
from src.utils.utils import AverageMeter, clean_message


class Trainer:
    def __init__(self, config: DictConfig, logger: loguru._Logger, save_suffix: str = ""):
        self.config = config
        self.logger = logger
        self.save_suffix = save_suffix
        self.detail_pbar = True

        self.model = ComponentFactory.get_model(config)
        self.model = self.model.to(config.device)
        n_device = torch.cuda.device_count()
        if n_device > 1:
            self.model = nn.DataParallel(self.model)

        if self.config.ema:
            self.model_ema = ModelEmaV3(
                self.model,
                decay=config.ema_decay,
                update_after_step=config.ema_update_after_step,
                device=config.device,
            )

        self.loss_fn = ComponentFactory.get_loss(config)
        self.train_loss = AverageMeter()
        self.valid_loss = AverageMeter()

        self.optimizer = None
        self.scheduler = None
        self.grad_scaler = amp.GradScaler(enabled=config.amp)

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader | None,
        retrain: bool = False,
        retrain_weight_name: str = "",
        retrain_best_score: float = -np.inf,
        full_train: bool = False,
        full_steps: int = 0,
        eval_only: bool = False,
    ):
        if eval_only:
            assert not full_train, "eval_only and full_train cannot be True at the same time"
            score, loss, oof_df = self.valid_evaluate(valid_loader, epoch=-1, load_best_weight=True)
            return score, -1, oof_df

        self.optimizer = ComponentFactory.get_optimizer(self.config, self.model)

        global_steps = 0
        update_steps = 0
        best_score = -np.inf

        if retrain:
            self.model.load_state_dict(torch.load(self.config.output_path / f"{retrain_weight_name}.pth"))
            self.model_ema.update_after_step = 0
            best_score = retrain_best_score

        # 学習ループの開始
        epochs = self.config.epochs if not retrain else self.config.add_epochs
        for epoch in tqdm(range(epochs)):
            self.model.train()
            self.train_loss.reset()

            # 1epoch目はbackboneをfreezeする
            if epoch == 0 and not retrain:
                self.model.freeze_backbone(config.reinit_layer_num)
            elif epoch == 1 and not retrain:
                self.model.unfreeze_backbone(config.freeze_layer_num)

            iterations = tqdm(train_loader, total=len(train_loader)) if self.detail_pbar else train_loader
            for data in iterations:
                _, loss = self.forward_step(self.model, data)
                self.train_loss.update(loss.item(), n=data[0].size(0))
                loss = loss / self.config.accumulation_steps
                self.grad_scaler.scale(loss).backward()
                global_steps += 1

                if global_steps % self.config.accumulation_steps == 0:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    self.optimizer.zero_grad()
                    update_steps += 1

                    if self.config.ema:
                        self.model_ema.update(self.model, update_steps)

                    # backboneの学習が始まってからschedulerを適用
                    if epoch >= 1 or retrain:
                        if self.scheduler is None:
                            first_cycle_epochs = (
                                self.config.first_cycle_epochs if not retrain else self.config.add_first_cycle_epochs
                            )
                            total_steps = first_cycle_epochs * len(train_loader)
                            if not retrain:
                                total_steps -= len(train_loader)  # 最初の1epoch分はstepしないから
                            self.scheduler = ComponentFactory.get_scheduler(
                                self.config, self.optimizer, total_steps=total_steps
                            )
                        self.scheduler.step()

                if global_steps % self.config.eval_steps == 0 and not full_train:
                    score, loss, oof_df = self.valid_evaluate(valid_loader, epoch, load_best_weight=False)
                    if score > best_score:
                        best_score = score
                        best_steps = global_steps
                        best_oof = oof_df
                        parameters = self.model_ema.module.state_dict() if self.config.ema else self.model.state_dict()
                        torch.save(
                            parameters,
                            self.config.output_path / f"model{self.save_suffix}_best.pth",
                        )
                    self.model.train()

                if full_train and global_steps >= full_steps:
                    parameters = self.model_ema.module.state_dict() if self.config.ema else self.model.state_dict()
                    torch.save(
                        parameters,
                        self.config.output_path / f"model{self.save_suffix}_full.pth",
                    )
                    return None

            message = f"""
                [Train] :
                    Epoch={epoch},
                    Loss={self.train_loss.avg:.5f},
                    LR={self.optimizer.param_groups[0]["lr"]:.5e}
            """
            self.logger.info(clean_message(message))

            if self.config.smooth_type == "online":
                self.loss_fn.update_soft_matrix()

        return best_score, best_steps, best_oof

    def valid_evaluate(self, valid_loader: DataLoader, epoch: int, load_best_weight: bool = False):
        if load_best_weight:
            self.model.load_state_dict(torch.load(self.config.output_path / f"model{self.save_suffix}_best.pth"))

        self.model.eval()
        preds = []
        with torch.no_grad():
            iterations = tqdm(valid_loader, total=len(valid_loader)) if self.detail_pbar else valid_loader
            for data in iterations:
                if load_best_weight or not self.config.ema:
                    out, loss = self.forward_step(self.model, data)
                else:
                    out, loss = self.forward_step(self.model_ema, data)

                self.valid_loss.update(loss.item(), n=data[0].size(0))
                preds.extend(F.softmax(out, dim=-1).cpu().numpy().tolist())

        oof_df = self.get_oof_df(preds, valid_loader)
        score, best_th = get_best_negative_threshold(self.config, oof_df)

        loss = self.valid_loss.avg
        message = f"""
            Valid :
                Epoch={epoch},
                Loss={loss:.5f},
                Score={score:.5f}
                Threshold={best_th}
        """
        self.logger.info(clean_message(message))
        return score, loss, oof_df

    def forward_step(self, model: nn.Module, data: torch.Tensor):
        input_ids, attention_mask, positions_feats, labels = data
        input_ids = input_ids.to(self.config.device)
        attention_mask = attention_mask.to(self.config.device)
        positions_feats = positions_feats.to(self.config.device)
        labels = labels.to(self.config.device)

        with amp.autocast(enabled=self.config.amp):
            out = model(input_ids, attention_mask, positions_feats)
            loss = self.loss_fn(out, labels)
        return out, loss

    def get_oof_df(self, preds: list[list[float]], valid_loader: DataLoader):
        char_pred_df = get_char_pred_df(
            preds,
            valid_loader.dataset.overlap_doc_ids,
            valid_loader.dataset.offset_mapping,
            class_num=self.config.class_num,
        )
        char2org_df = get_char2org_df(
            valid_loader.dataset.doc_ids,
            valid_loader.dataset.full_texts,
            valid_loader.dataset.org_tokens,
            valid_loader.dataset.whitespaces,
        )
        oof_df = char_pred_df.join(char2org_df, on=["document", "char_idx"], how="left", coalesce=True)
        oof_df = (
            oof_df.filter(pl.col("token_idx") != -1)
            .group_by("document", "token_idx")
            .agg([pl.col(f"pred_{i}").mean() for i in range(self.config.class_num)])
        )
        return oof_df


# # Run

# In[10]:


# oof_dfs = []
# best_steps, best_add_steps = [], []
# collate_fn = CollateFn(get_tokenizer(config), is_train=True)

# # この学習でベストなステップ数とOOFに対する予測値を取ることが目的
# for fold, (train_loader, valid_loader) in enumerate(dataloaders):
#     logger.info(f"\n FOLD{fold} : Training Start \n")

#     # First Training
#     trainer = Trainer(config, logger, save_suffix=f"_fold{fold}")
#     best_score, best_steps_, _ = trainer.train(train_loader, valid_loader)
#     if config.smooth_type == "online":
#         loss_soft_matrix = trainer.loss_fn.soft_matrix
#     best_steps.append(best_steps_)
#     logger.info(f"\n FOLD{fold} : First Training Done! -->> Best Score: {best_score}, Best Steps: {best_steps_} \n")

#     del trainer
#     gc.collect()
#     torch.cuda.empty_cache()

#     # Create High-Quality Dataloader
#     train_dataset = train_loader.dataset
#     train_dataset.drop_first_only_data()
#     train_loader = DataLoader(
#         train_dataset,
#         sampler=get_sampler(train_dataset),
#         batch_size=config.train_batch,
#         collate_fn=collate_fn,
#         pin_memory=True,
#         drop_last=True,
#     )

#     # Additional Training
#     trainer = Trainer(config, logger, save_suffix=f"_fold{fold}")
#     if config.smooth_type == "online":
#         trainer.loss_fn.soft_matrix = loss_soft_matrix.clone()
#     best_score, best_add_steps_, oof_df = trainer.train(
#         train_loader,
#         valid_loader,
#         retrain=True,
#         retrain_weight_name=f"model_fold{fold}_best",
#         retrain_best_score=best_score,
#     )
#     best_add_steps.append(best_add_steps_)
#     oof_df.write_parquet(config.output_path / f"oof_fold{fold}.parquet")
#     oof_dfs.append(oof_df)
#     logger.info(
#         f"\n FOLD{fold} : Additional Training Done! -->> Best Score: {best_score}, Best Add Steps: {best_add_steps_} \n"
#     )

#     del train_loader, valid_loader, train_dataset, trainer, oof_df
#     gc.collect()
#     torch.cuda.empty_cache()

# del dataloaders
# gc.collect()

# # Save OOF
# oof_df = pl.concat(oof_dfs)
# oof_df.write_parquet(config.output_path / "oof.parquet")
# del oof_dfs
# gc.collect()

# # Get Best Negative Threshold
# best_score, best_th = get_best_negative_threshold(config, oof_df)
# message = f"Overall OOF Best Score: {best_score}, Best Negative Threshold: {best_th}"
# logger.info(message)
# config.negative_th = best_th.item()


# In[11]:


# これなら通る
# best_steps = [10, 5, 5]
# best_add_steps = [10, 5, 5]

# これだと落ちる -> メモリーエラーではない
N = 100
best_steps = [N, N, N]
best_add_steps = [N, N, N]

collate_fn = CollateFn(get_tokenizer(config), is_train=True)


# In[12]:


# # 全データ学習を行う
if config.full_train:
    full_steps = np.max(best_steps)
    full_add_steps = np.max(best_add_steps)
    logger.info("\n Full Train : Training Start \n")
    train_loader = get_full_train_loader(config, data)

    # First Training
    trainer = Trainer(config, logger, save_suffix="")
    trainer.train(train_loader, valid_loader=None, full_train=True, full_steps=full_steps)
    if config.smooth_type == "online":
        loss_soft_matrix = trainer.loss_fn.soft_matrix
    logger.info("\n Full Train : First Training Done! \n")

    # trainer.model.to("cpu")
    # # trainer.model_ema.to("cpu")
    # del trainer.model, trainer
    # gc.collect()
    # torch.cuda.empty_cache()

    print("hello world after model delete")

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

    # # Additional Training
    # trainer = Trainer(config, logger, save_suffix="")
    # if config.smooth_type == "online":
    #     trainer.loss_fn.soft_matrix = loss_soft_matrix.clone()
    # trainer.train(
    #     train_loader,
    #     valid_loader=None,
    #     retrain=True,
    #     retrain_weight_name="model_full",
    #     full_train=True,
    #     full_steps=full_add_steps,
    # )
    # logger.info("\n Full Train : Additional Training Done! \n")

    # del train_loader, trainer
    # gc.collect()
    # torch.cuda.empty_cache()


class Trainer:
    def __init__(self, config: DictConfig, logger: loguru._Logger, save_suffix: str = ""):
        print("hello world")
        # self.config = config
        # self.logger = logger
        # self.save_suffix = save_suffix
        # self.detail_pbar = True

        # self.model = ComponentFactory.get_model(config)
        # self.model = self.model.to(config.device)

        # if self.config.ema:
        #     self.model_ema = ModelEmaV3(
        #         self.model,
        #         decay=config.ema_decay,
        #         update_after_step=config.ema_update_after_step,
        #         device=config.device,
        #     )

        # self.loss_fn = ComponentFactory.get_loss(config)
        # self.train_loss = AverageMeter()
        # self.valid_loss = AverageMeter()

        # self.optimizer = None
        # self.scheduler = None
        # self.grad_scaler = amp.GradScaler(enabled=config.amp)

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader | None,
        retrain: bool = False,
        retrain_weight_name: str = "",
        retrain_best_score: float = -np.inf,
        full_train: bool = False,
        full_steps: int = 0,
        eval_only: bool = False,
    ):
        if eval_only:
            assert not full_train, "eval_only and full_train cannot be True at the same time"
            score, loss, oof_df = self.valid_evaluate(valid_loader, epoch=-1, load_best_weight=True)
            return score, -1, oof_df

        self.optimizer = ComponentFactory.get_optimizer(self.config, self.model)

        global_steps = 0
        update_steps = 0
        best_score = -np.inf

        if retrain:
            self.model.load_state_dict(torch.load(self.config.output_path / f"{retrain_weight_name}.pth"))
            self.model_ema.update_after_step = 0
            best_score = retrain_best_score

        # 学習ループの開始
        epochs = self.config.epochs if not retrain else self.config.add_epochs
        for epoch in tqdm(range(epochs)):
            self.model.train()
            self.train_loss.reset()

            # 1epoch目はbackboneをfreezeする
            if epoch == 0 and not retrain:
                self.model.freeze_backbone(config.reinit_layer_num)
            elif epoch == 1 and not retrain:
                self.model.unfreeze_backbone(config.freeze_layer_num)

            iterations = tqdm(train_loader, total=len(train_loader)) if self.detail_pbar else train_loader
            for data in iterations:
                _, loss = self.forward_step(self.model, data)
                self.train_loss.update(loss.item(), n=data[0].size(0))
                loss = loss / self.config.accumulation_steps
                self.grad_scaler.scale(loss).backward()
                global_steps += 1

                if global_steps % self.config.accumulation_steps == 0:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    self.optimizer.zero_grad()
                    update_steps += 1

                    if self.config.ema:
                        self.model_ema.update(self.model, update_steps)

                    # backboneの学習が始まってからschedulerを適用
                    if epoch >= 1 or retrain:
                        if self.scheduler is None:
                            first_cycle_epochs = (
                                self.config.first_cycle_epochs if not retrain else self.config.add_first_cycle_epochs
                            )
                            total_steps = first_cycle_epochs * len(train_loader)
                            if not retrain:
                                total_steps -= len(train_loader)  # 最初の1epoch分はstepしないから
                            self.scheduler = ComponentFactory.get_scheduler(
                                self.config, self.optimizer, total_steps=total_steps
                            )
                        self.scheduler.step()

                if global_steps % self.config.eval_steps == 0 and not full_train:
                    score, loss, oof_df = self.valid_evaluate(valid_loader, epoch, load_best_weight=False)
                    if score > best_score:
                        best_score = score
                        best_steps = global_steps
                        best_oof = oof_df
                        parameters = self.model_ema.module.state_dict() if self.config.ema else self.model.state_dict()
                        torch.save(
                            parameters,
                            self.config.output_path / f"model{self.save_suffix}_best.pth",
                        )
                    self.model.train()

                if full_train and global_steps >= full_steps:
                    parameters = self.model_ema.module.state_dict() if self.config.ema else self.model.state_dict()
                    torch.save(
                        parameters,
                        self.config.output_path / f"model{self.save_suffix}_full.pth",
                    )
                    return None

            message = f"""
                [Train] :
                    Epoch={epoch},
                    Loss={self.train_loss.avg:.5f},
                    LR={self.optimizer.param_groups[0]["lr"]:.5e}
            """
            self.logger.info(clean_message(message))

            if self.config.smooth_type == "online":
                self.loss_fn.update_soft_matrix()

        return best_score, best_steps, best_oof

    def valid_evaluate(self, valid_loader: DataLoader, epoch: int, load_best_weight: bool = False):
        if load_best_weight:
            self.model.load_state_dict(torch.load(self.config.output_path / f"model{self.save_suffix}_best.pth"))

        self.model.eval()
        preds = []
        with torch.no_grad():
            iterations = tqdm(valid_loader, total=len(valid_loader)) if self.detail_pbar else valid_loader
            for data in iterations:
                if load_best_weight or not self.config.ema:
                    out, loss = self.forward_step(self.model, data)
                else:
                    out, loss = self.forward_step(self.model_ema, data)

                self.valid_loss.update(loss.item(), n=data[0].size(0))
                preds.extend(F.softmax(out, dim=-1).cpu().numpy().tolist())

        oof_df = self.get_oof_df(preds, valid_loader)
        score, best_th = get_best_negative_threshold(self.config, oof_df)

        loss = self.valid_loss.avg
        message = f"""
            Valid :
                Epoch={epoch},
                Loss={loss:.5f},
                Score={score:.5f}
                Threshold={best_th}
        """
        self.logger.info(clean_message(message))
        return score, loss, oof_df

    def forward_step(self, model: nn.Module, data: torch.Tensor):
        input_ids, attention_mask, positions_feats, labels = data
        input_ids = input_ids.to(self.config.device)
        attention_mask = attention_mask.to(self.config.device)
        positions_feats = positions_feats.to(self.config.device)
        labels = labels.to(self.config.device)

        with amp.autocast(enabled=self.config.amp):
            out = model(input_ids, attention_mask, positions_feats)
            loss = self.loss_fn(out, labels)
        return out, loss

    def get_oof_df(self, preds: list[list[float]], valid_loader: DataLoader):
        char_pred_df = get_char_pred_df(
            preds,
            valid_loader.dataset.overlap_doc_ids,
            valid_loader.dataset.offset_mapping,
            class_num=self.config.class_num,
        )
        char2org_df = get_char2org_df(
            valid_loader.dataset.doc_ids,
            valid_loader.dataset.full_texts,
            valid_loader.dataset.org_tokens,
            valid_loader.dataset.whitespaces,
        )
        oof_df = char_pred_df.join(char2org_df, on=["document", "char_idx"], how="left", coalesce=True)
        oof_df = (
            oof_df.filter(pl.col("token_idx") != -1)
            .group_by("document", "token_idx")
            .agg([pl.col(f"pred_{i}").mean() for i in range(self.config.class_num)])
        )
        return oof_df


# Additional Training
trainer = Trainer(config, logger, save_suffix="")
# if config.smooth_type == "online":
#     trainer.loss_fn.soft_matrix = loss_soft_matrix.clone()


# In[ ]:


# trainer


# # In[ ]:


# # Additional Training
# trainer = Trainer(config, logger, save_suffix="")
# # if config.smooth_type == "online":
# #     trainer.loss_fn.soft_matrix = loss_soft_matrix.clone()


# # # Test

# # In[ ]:


# import re

# import numpy as np
# import polars as pl
# from tqdm import tqdm

# from src.data.load import load_competition_data
# from src.data.utils import get_original_token_df, get_truth_df
# from src.utils.constant import IDX2TARGET_WITH_BIO, TARGET2IDX_WITH_BIO


# class PostProcessor:
#     """
#     後処理を行うクラス

#     後処理一覧
#         - 1. 空白文字に対してPositiveと予測している場合に、そのトークンの予測を"O"(idx=0)に置き換える
#         - 2. ラベルの妥当性を確保する
#             - 2-1. B, Bが連続していて、かつ同一のタイプである場合に、後続のBをIに変更する
#             - 2-2. Iの前がOで、かつ閾値以内に同一タイプのB, Iが存在する場合に、間のOをIに変更する
#             - 2-3. Iの前がOで、かつ2-2の場合でない場合に、IをBに変更する
#         - 3. ラベルのタイプが混在している場合に、それを1つのラベルに統合して、さらにFPを弾く処理を行う

#     Expected DataFrame
#         - pred_df (pl.DataFrame): [document, token_index, prob, pred]
#             - 全てのトークンに対しての予測を含める必要がある("O"の場合も除去しない)
#     """

#     def __init__(self, config: DictConfig):
#         self.config = config
#         self.org_data = load_json_data(config.input_path / f"{config.run_type}.json", debug=config.debug)
#         self.full_texts = [d["full_text"] for d in self.org_data]

#     def post_process(self, pred_df: pl.DataFrame) -> pl.DataFrame:
#         pred_df = pred_df.with_columns(
#             pred_org=pl.col("pred").replace(IDX2TARGET_WITH_BIO, default=""),
#         )
#         pred_doc_ids = pred_df["document"].unique().to_numpy()

#         org_token_df = get_original_token_df(self.config, pred_doc_ids)
#         pred_df = pred_df.join(org_token_df, on=["document", "token_idx"], how="left", coalesce=True)

#         if self.config.run_type == "train":
#             truth_df = get_truth_df(config, pred_doc_ids, convert_idx=True)
#             truth_df = truth_df.with_columns(label_org=pl.col("label").repclace(IDX2TARGET_WITH_BIO, default=""))
#             pred_df = pred_df.join(truth_df, on=["document", "token_idx"], how="left", coalesce=True)

#         pred_df = pred_df.with_columns(pred_org_prev=pl.col("pred_org"))

#         # 1. 空白文字をPositiveと予測している場合に、そのトークンの予測を0("O")に置き換える
#         pred_df = self.remove_space_pii(pred_df)

#         # # 2. ラベルの妥当性を確保する
#         # pred_df = self.check_bio_validity(pred_df)
#         # # 2の処理で空白に再度PIIが予測される場合があるため、再度1の処理を行う
#         # pred_df = self.remove_space_pii(pred_df)

#         # # 3. ラベルのタイプが混在している場合に、それを1つのラベルに統合して、さらにFPを弾く処理を行う
#         # pred_df = self.check_pii_validity(pred_df)

#         # return pred_df

#     def remove_space_pii(self, pred_df: pl.DataFrame):
#         pred_df = pred_df.with_columns(
#             pred=(
#                 pl.when(pl.col("token").map_elements(lambda x: re.sub(r"[ \xa0]+", " ", x)) == " ")
#                 .then(pl.lit(0))
#                 .otherwise(pl.col("pred"))
#             ),
#             pred_org=(
#                 pl.when(pl.col("token").map_elements(lambda x: re.sub(r"[ \xa0]+", " ", x)) == " ")
#                 .then(pl.lit("O"))
#                 .otherwise(pl.col("pred_org"))
#             ),
#         )
#         return pred_df

#     def get_prev_pred_label(self, preds_array: np.ndarray, idx: int):
#         if idx >= 0:
#             label = preds_array[idx]
#             if label == "O":
#                 return "O", None
#             else:
#                 return label.split("-")
#         else:
#             return None, None

#     def check_bio_validity(self, pred_df: pl.DataFrame, continue_th: int = 1):
#         """
#         1. B, Bが連続していて、かつ同一のタイプである場合に、後続のBをIに変更する
#         2. Iの前がOで、かつ閾値以内に同一タイプのB, Iが存在する場合に、間のOをIに変更する
#         3. Iの前がOで、かつ2の場合でない場合に、IをBに変更する
#         """
#         pred_pp = []
#         for _, doc_df in tqdm(pred_df.group_by("document"), total=pred_df["document"].n_unique()):
#             doc_df = doc_df.sort("token_index")
#             preds_org = doc_df["pred_org"].to_numpy()
#             new_preds_org = preds_org.copy()  # 後処理後の予測ラベル

#             for i in range(len(doc_df)):
#                 if preds_org[i] == "O":  # Oの場合はスキップ
#                     continue

#                 prefix_c, label_type_c = preds_org[i].split("-")
#                 prefix_p1, label_type_p1 = self.get_prev_pred_label(preds_org, i - 1)  # 1つ前のラベルを取得

#                 # 1. B, Bが連続していて、かつ同一のタイプである場合に、後続のBをIに変更する
#                 if prefix_c == "B" and prefix_p1 == "B" and label_type_c == label_type_p1:
#                     new_preds_org[i] = f"I-{label_type_c}"

#                 # 2. Iの前がOで、かつ閾値以内に同一タイプのB, Iが存在する場合に、間のOをIに変更する
#                 if prefix_c == "I" and prefix_p1 not in ["B", "I"]:
#                     # 閾値以内に同一タイプのB, Iが存在するかどうか
#                     find_flag = False
#                     # for j in range(2, continue_th + 2):
#                     #     prefix_p, label_type_p = self.get_prev_pred_label(new_preds_org, i-j)
#                     #     if prefix_p in ['B', 'I'] and label_type_c == label_type_p:
#                     #         find_flag = True
#                     #         for k in range(1, j):
#                     #             new_preds_org[i-k] = f'I-{label_type_c}' # 閾値以内に同一タイプのB, Iが存在する場合はOをIに変更

#                     # 3. Iの前がOで、かつ2の場合でない場合に、IをBに変更する
#                     if find_flag == False:
#                         new_preds_org[i] = f"B-{label_type_c}"

#             doc_df = doc_df.with_columns(pred_org=pl.Series(new_preds_org.tolist()).cast(pl.Utf8)).with_columns(
#                 pred=pl.col("pred_org").replace(TARGET2IDX_WITH_BIO, default=0)
#             )
#             pred_pp.append(doc_df)

#         pred_pp = pl.concat(pred_pp)
#         return pred_pp

#     def get_pii_string(self, pii_df: pl.DataFrame):
#         pii_string = ""
#         for token, space in pii_df[["token", "space"]].to_numpy():
#             if space:
#                 pii_string += token + " "
#             else:
#                 pii_string += token
#         return pii_string.strip()

#     def remove_token_base(self, pii_df: pl.DataFrame, char_len: int):
#         pii_df = pii_df.with_columns(
#             pred=pl.when(pl.col("token").map_elements(lambda x: len(x.strip()) <= char_len))
#             .then(pl.lit(0))
#             .otherwise(pl.col("pred")),  # char_len以下の文字数のトークンを除去
#             pred_org=pl.when(pl.col("token").map_elements(lambda x: len(x.strip()) <= char_len))
#             .then(pl.lit("O"))
#             .otherwise(pl.col("pred_org")),
#         )
#         return pii_df

#     def remove_pii_base(self, pii_df: pl.DataFrame):
#         pii_df = pii_df.with_columns(pred=pl.lit(0).cast(pl.Int64), pred_org=pl.lit("O"))
#         return pii_df

#     def remove_false_positive(self, pii_df: pl.DataFrame):
#         pii_type = pii_df["pii_type"][0]
#         pii_string = self.get_pii_string(pii_df)

#         if pii_type == "NAME_STUDENT":
#             # トークンの先頭の文字が大文字でない場合かつ,1文字の場合に除去
#             pii_df = pii_df.with_columns(
#                 pred=pl.when(pl.col("token").map_elements(lambda x: (x[0].isupper()) and (len(x.strip()) > 1)))
#                 .then(pl.col("pred"))
#                 .otherwise(pl.lit(0)),
#                 pred_org=pl.when(pl.col("token").map_elements(lambda x: (x[0].isupper()) and (len(x.strip()) > 1)))
#                 .then(pl.col("pred_org"))
#                 .otherwise(pl.lit("O")),
#             )

#         elif pii_type == "EMAIL":
#             # 下記のフォーマットでマッチしない場合は除去する
#             email_pattern = r"[^@ \t\r\n]+@[^@ \t\r\n]+\.(?:com|org|edu)"  # [任意の文字列]@[任意の文字列].com|org|edu
#             if re.search(email_pattern, pii_string) is None:
#                 pii_df = self.remove_pii_base(pii_df)
#                 return pii_df

#         elif pii_type == "USERNAME":
#             # トークンが1文字の場合に除去
#             pii_df = self.remove_token_base(pii_df, char_len=1)

#         elif pii_type == "ID_NUM":
#             # トークンが1文字の場合に除去
#             pii_df = self.remove_token_base(pii_df, char_len=1)
#             # N桁未満の数字のみの場合に除去
#             num_only = re.search(r"^\d+$", pii_string)
#             if num_only is not None and len(pii_string) < 4:  # 4桁未満の数字のみの場合に除去
#                 pii_df = self.remove_pii_base(pii_df)
#                 return pii_df

#         elif pii_type == "PHONE_NUM":
#             # 数字以外を除去して数字がN桁(電話番号の最小桁数は10桁)未満の場合は除去する -> 全体を取りきれていないケースも存在しそうなのでFNを取り切るようにした方がいいかも
#             num_string = re.sub(r"\D+", "", pii_string)
#             if len(num_string) < 10:
#                 pii_df = self.remove_pii_base(pii_df)
#                 return pii_df

#         # elif pii_type == 'URL_PERSONAL':
#         #     # # 下記のフォーマットでマッチしない場合は除去する -> 他のprefixも追加する
#         #     # url_pattern = r'^(?:http|https)://' # http://, https://で始まるURL
#         #     # if re.search(url_pattern, pii_string) is None:
#         #     #     pii_df = self.remove_pii_base(pii_df)
#         #     #     return pii_df

#         #     # 以下を含む場合は除去する
#         #     remove_target = [
#         #         'wikipedia.org',
#         #     ]
#         #     if any([target in pii_string for target in remove_target]):
#         #         pii_df = self.remove_pii_base(pii_df)
#         #         return pii_df

#         elif pii_type == "STREET_ADDRESS":
#             # 10文字未満であれば除去する -> もっと桁数を増やしてもいいかもしれない
#             if len(pii_string) < 10:
#                 pii_df = self.remove_pii_base(pii_df)
#                 return pii_df

#         # # pii_stringが出現するテキストのユニークな数をカウントする -> 一定以上の生徒のテキストで出現する場合は除去する
#         # unique_appear_count = sum([1 for text in self.full_text_list if pii_string in text])
#         # condition = (
#         #     ((unique_appear_count >= 2) and (pii_type != 'NAME_STUDENT'))
#         #     # ((unique_appear_count >= 4) and (pii_type == 'NAME_STUDENT') and len(pii_df) >= 2)
#         # )
#         # if condition:
#         #     pii_df = self.remove_pii_base(pii_df)
#         return pii_df

#     def check_pii_validity(self, pred_df: pl.DataFrame):
#         # 各PIIに固有のグループIDを割り当てる
#         pred_df = pred_df.with_row_index(name="pii_group").with_columns(
#             pii_prefix=pl.col("pred_org").map_elements(lambda x: x if x == "O" else x.split("-")[0])
#         )
#         pred_df = pred_df.with_columns(
#             pii_group=(
#                 pl.when(pl.col("pii_prefix") == "I")
#                 .then(pl.lit(None))
#                 .when(pl.col("pii_prefix") == "O")
#                 .then(pl.lit(-1))
#                 .otherwise(pl.col("pii_group"))
#             )
#         )
#         pred_df = pred_df.sort(["document", "token_index"])
#         pred_df = pred_df.with_columns(pii_group=pl.col("pii_group").fill_null(strategy="forward").over("document"))

#         # 一度pandasに変換して再度polarsに変換を行う -> これを行わないと激遅 (原因不明, メモリの配置やソートの問題ではないっぽい)
#         pred_df = pred_df.to_pandas()
#         pred_df = pl.from_pandas(pred_df)

#         pred_pp = []
#         for (_, pii_group), pii_df in tqdm(
#             pred_df.group_by(["document", "pii_group"]), total=len(pred_df.unique(subset=["document", "pii_group"]))
#         ):
#             if pii_group == -1:
#                 pred_pp.append(pii_df)
#                 continue

#             # Bから始まり、それ以外はIであることを確認
#             for i, prefix in enumerate(pii_df["pii_prefix"].to_list()):
#                 if i == 0:
#                     assert prefix == "B"
#                 else:
#                     assert prefix == "I"

#             pii_df = pii_df.with_columns(
#                 pii_type=pl.col("pred_org").map_elements(lambda x: x.split("-")[1] if x != "O" else None)
#             )

#             # PIIタイプが異なる場合は、最も確率が高いタイプに統一する
#             if pii_df["pii_type"].n_unique() > 1:
#                 highest_type = (
#                     pii_df.group_by("pii_type").agg(pl.col("prob").sum()).sort("prob", descending=True)["pii_type"][0]
#                 )
#                 pii_df = pii_df.with_columns(
#                     pred_org=pl.concat_str([pl.col("pii_prefix"), pl.lit("-"), pl.lit(highest_type)])
#                 )
#                 pii_df = pii_df.with_columns(pred=pl.col("pred_org").replace(TARGET2IDX_WITH_BIO, default=0))

#             # # FPを弾く処理を行う
#             pii_df = self.remove_false_positive(pii_df)
#             pred_pp.append(pii_df)

#         # 全体を結合する
#         pred_pp = (
#             pl.concat(pred_pp, how="diagonal")
#             .sort("document", "token_index")
#             .drop(["pii_group", "pii_prefix", "pii_type"])
#         )
#         return pred_pp


# # In[ ]:


# # 後処理で精度がどれくらい変わるのかを確認
# oof_df = pl.read_parquet(config.oof_path / f"oof_{config.exp}{suffix}.parquet")
# pred_df = get_pred_df(oof_df, negative_th=config.negative_th)

# pper = PostProcessor(config, pred_df)
# pred_df = pper.get_post_processed_df()

# score = evaluate_metric(pred_df, get_truth_df(config, pred_df["document"].unique().to_list(), is_label_idx=True))
# print(f"Post Processed Score: {score:.4f}")
# logger.info(f"Post Processed Score: {score:.4f}")


# # In[ ]:
