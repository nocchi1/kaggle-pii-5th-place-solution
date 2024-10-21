import loguru
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.train.component_factory import ComponentFactory
from src.train.ema import ModelEmaV3
from src.utils.competition_utils import get_char2org_df, get_char_pred_df
from src.utils.metric import get_best_negative_threshold
from src.utils.utils import AverageMeter, clean_message


class Trainer:
    def __init__(self, config: DictConfig, logger: loguru._Logger, save_suffix: str = ""):
        self.config = config
        self.logger = logger
        self.save_suffix = save_suffix
        self.detail_pbar = True

        self.model = ComponentFactory.get_model(config)
        self.model = self.model.to(config.device)

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
        best_steps = 0
        best_oof = None
        full_train_complete = False

        if retrain:
            self.model.load_state_dict(torch.load(self.config.output_path / f"{retrain_weight_name}.pth"))
            self.model_ema.update_after_step = 0
            best_score = retrain_best_score

        # 学習ループの開始
        epochs = self.config.epochs if not retrain else self.config.add_epochs
        for epoch in tqdm(range(epochs)):
            if full_train_complete:
                break

            self.model.train()
            self.train_loss.reset()

            # 1epoch目はbackboneをfreezeする
            if epoch == 0 and not retrain:
                self.model.freeze_backbone(self.config.reinit_layer_num)
            elif epoch == 1 and not retrain:
                self.model.unfreeze_backbone(self.config.freeze_layer_num)

            iterations = tqdm(train_loader, total=len(train_loader)) if self.detail_pbar else train_loader
            for data in iterations:
                if full_train_complete:  # ループの上部でbreakする必要がある
                    break

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

                if full_train and global_steps == full_steps:
                    parameters = self.model_ema.module.state_dict() if self.config.ema else self.model.state_dict()
                    torch.save(
                        parameters,
                        self.config.output_path / f"model{self.save_suffix}_full.pth",
                    )
                    full_train_complete = True  # ここでbreakすると何故かnotebook kernelが落ちる現象が発生する

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
