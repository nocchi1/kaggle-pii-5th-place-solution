from typing import Dict, List

import numpy as np
import polars as pl
from tqdm import tqdm

from src.utils.competition_utils import get_pred_df, get_truth_df, restore_prefix


def calculate_fbeta(tp: int, fp: int, fn: int, beta: float = 5.0):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    if precision == 0 and recall == 0:
        return 0.0
    score = (1 + (beta**2)) * precision * recall / (beta**2 * precision + recall)
    return score


def evaluate_metric(
    pred_df: pl.DataFrame,
    truth_df: pl.DataFrame,
) -> float:
    truth_df = truth_df.join(pred_df, on=["document", "token_index"], how="left")
    truth_df = truth_df.with_columns(pred=pl.col("pred").fill_null(0))

    tp = len(truth_df.filter((pl.col("label") != 0) & (pl.col("label") == pl.col("pred"))))
    fp = len(truth_df.filter((pl.col("label") == 0) & (pl.col("pred") != 0)))
    fn = len(truth_df.filter((pl.col("label") != 0) & (pl.col("pred") == 0)))
    fp_fn = len(truth_df.filter((pl.col("label") != 0) & (pl.col("pred") != 0) & (pl.col("label") != pl.col("pred"))))
    score = calculate_fbeta(tp, fp + fp_fn, fn + fp_fn)
    return score


# def get_best_negative_threshold(config, oof_df: pl.DataFrame):
#     truth_df = get_truth_df(config, oof_df["document"].unique().to_list(), is_label_idx=True)
#     best_score = 0
#     lower_th, upper_th = 0.20, 1.00
#     for th in tqdm(np.arange(lower_th, upper_th, 0.01)):
#         pred_df = get_pred_df(oof_df, negative_th=th)
#         if config.remove_prefix:
#             pred_df = restore_prefix(config, pred_df)
#         score = evaluate_metric(pred_df, truth_df)
#         if score > best_score:
#             best_score = score
#             best_th = th
#     return best_score, best_th


# def get_best_negative_individual_threshold(
#     config, oof_df: pl.DataFrame
# ) -> dict[int, float]:  # ここのpii_typeごとのthresholdを返す
#     # 最も確率の高いpositiveクラスを算出
#     pred_df = oof_df.with_columns(
#         positive_pred=np.argmax(
#             oof_df.select([col for col in oof_df.columns if "pred" in col and "pred_0" != col]).to_numpy(),
#             axis=1,
#         )
#         + 1,
#         positive_prob=np.max(
#             oof_df.select([col for col in oof_df.columns if "pred" in col and "pred_0" != col]).to_numpy(),
#             axis=1,
#         ),
#         negative_prob=pl.col("pred_0"),
#     )
#     # Truthを取得して結合
#     truth_df = get_truth_df(config, oof_df["document"].unique().to_list(), is_label_idx=True)
#     pred_df = pred_df.join(truth_df, on=["document", "token_index"], how="left")
#     # 各pii_typeごとにnegative thresholdを算出
#     best_th_dict = {}
#     lower_th, upper_th = 0.00, 1.00
#     for pii_type, pii_pred_df in tqdm(pred_df.group_by(["positive_pred"]), total=pred_df["positive_pred"].n_unique()):
#         best_score = 0
#         for th in tqdm(np.arange(lower_th, upper_th, 0.01)):
#             pii_pred_df = pii_pred_df.with_columns(
#                 prob=(
#                     pl.when(pl.col("negative_prob") > th)
#                     .then(pl.col("negative_prob"))
#                     .otherwise(pl.col("positive_prob"))
#                 ).alias("prob"),
#                 pred=(
#                     pl.when(pl.col("negative_prob") > th)
#                     .then(pl.lit(0).cast(pl.Int64))
#                     .otherwise(pl.col("positive_pred"))
#                 ).alias("pred"),
#             )
#             # 後で共通化する
#             tp = pii_pred_df.filter((pl.col("label") != 0) & (pl.col("label") == pl.col("pred"))).shape[0]
#             fp = pii_pred_df.filter((pl.col("label") == 0) & (pl.col("pred") != 0)).shape[0]
#             fn = pii_pred_df.filter((pl.col("label") != 0) & (pl.col("pred") == 0)).shape[0]
#             fp_fn = pii_pred_df.filter(
#                 (pl.col("label") != 0) & (pl.col("pred") != 0) & (pl.col("label") != pl.col("pred"))
#             ).shape[0]
#             score = calculate_fbeta(tp, fp + fp_fn, fn + fp_fn)
#             if score > best_score:
#                 best_score = score
#                 best_th = th
#         best_th_dict[pii_type[0]] = best_th
#     return best_th_dict


# def get_best_threshold_2nd(config, pred_df: pl.DataFrame):
#     truth_df = get_truth_df(config, pred_df["document"].unique().to_list(), is_label_idx=True)
#     best_score = 0
#     lower_th, upper_th = 0.000, 0.601
#     for th in np.arange(lower_th, upper_th, 0.005):
#         pred_df = pred_df.with_columns(
#             pred=(
#                 pl.when(
#                     (pl.col("pred_first").is_in([1, 8]))
#                     & (pl.col("pred_name") <= th)
#                     & (pl.col("pred_name").is_not_null())
#                 )
#                 .then(0)
#                 .otherwise(pl.col("pred_first"))
#             )
#         )
#         score = evaluate_metric(pred_df, truth_df)
#         if score > best_score:
#             best_score = score
#             best_th = th
#     return best_score, best_th


# def check_data_validity_2nd(train_data: list[dict], use_fn_label: bool):
#     """
#     2ndステージのラベルにおいて、labels, name_predsの関係からsecond_labelsが正しく作成できているのかを確認
#     TP, FP, FNの数も同時に確認する
#     """
#     tp, fp, fn = 0, 0, 0
#     for data in train_data:
#         assert len(data["labels"]) == len(data["name_preds"])
#         assert len(data["labels"]) == len(data["second_labels"])

#         for label, pred, second_label in zip(data["labels"], data["name_preds"], data["second_labels"]):
#             if pred == 1 and label in [1, 8]:
#                 tp += 1
#                 assert second_label == 1
#             elif pred == 1 and label not in [1, 8]:
#                 fp += 1
#                 assert second_label == 0
#             elif pred == 0 and label in [1, 8]:
#                 fn += 1
#                 if use_fn_label:
#                     assert second_label == 1
#                 else:
#                     assert second_label == -1
#             else:
#                 assert second_label == -1
#     return tp, fp, fn
