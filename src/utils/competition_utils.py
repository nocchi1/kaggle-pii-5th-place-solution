import json
import re
import statistics
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig

from src.utils.constant import IDX2TARGET_WTO_BIO, TARGET2IDX_WITH_BIO, TARGET2IDX_WTO_BIO


def load_json_data(file_path: Path, debug: bool = False) -> list[dict]:
    with open(file_path) as f:
        data = json.load(f)
    if debug:
        data = data[:100]
    return data


def remove_target_prefix(data: list[dict]) -> list[dict]:
    for i in range(len(data)):
        labels = data[i]["labels"]
        data[i]["labels"] = [re.sub(r"(B-|I-)", "", label) for label in labels]
    return data


def convert_label_str2index(data: list[dict], remove_prefix: bool = False) -> list[dict]:
    if remove_prefix:
        data = remove_target_prefix(data)

    target2idx = TARGET2IDX_WTO_BIO if remove_prefix else TARGET2IDX_WITH_BIO
    for i in range(len(data)):
        labels = data[i]["labels"]
        data[i]["labels"] = [target2idx[label] for label in labels]
    return data


def mapping_index_org2char(
    full_texts: list[str],
    org_tokens: list[list[str]],
    org_tokens_idx: list[list[int]],
    whitespaces: list[list[bool]],
    fill_val: int = -1,
) -> list[np.ndarray]:
    """
    Map the index of the original token to each character position in the text.
    """
    char_indices = []
    for i in range(len(full_texts)):
        text = full_texts[i]
        org_token = org_tokens[i]
        org_token_idx = org_tokens_idx[i]
        whitespace = whitespaces[i]

        start_idx = 0
        char_idx = np.full(len(text), fill_val, dtype=np.int32)
        for i, t, s in zip(org_token_idx, org_token, whitespace):
            char_idx[start_idx : start_idx + len(t)] = i
            start_idx += len(t) + int(s)
        char_indices.append(char_idx)
    return char_indices


def mapping_index_org2char(
    full_texts: list[str],
    org_tokens: list[list[str]],
    org_tokens_idx: list[list[int]],
    whitespaces: list[list[bool]],
    fill_val: int = -1,
):
    """
    Map the index of the original token to each character position in the text.
    """
    char_indices = []
    for i in range(len(full_texts)):
        text = full_texts[i]
        org_token = org_tokens[i]
        org_token_idx = org_tokens_idx[i]
        whitespace = whitespaces[i]

        start_idx = 0
        char_idx = np.full(len(text), fill_val, dtype=np.int32)
        for i, t, s in zip(org_token_idx, org_token, whitespace):
            char_idx[start_idx : start_idx + len(t)] = i
            start_idx += len(t) + int(s)
        char_indices.append(char_idx)
    return char_indices


def mapping_index_char2token(
    char_indices: list[list[int]],
    input_ids: list[list[int]],
    offset_mappings: list[tuple[int, int]],
    fill_val: int = -1,
):
    """
    Map the index of the original token to the token position in the model, based on character position.
    """
    token_indices = []
    for i, (ids, maps) in enumerate(zip(input_ids, offset_mappings)):
        assert len(ids) == len(maps)
        char_idx = char_indices[i]
        token_idx = np.zeros(len(ids), dtype=np.int32)
        for idx, (start, end) in enumerate(maps):
            if start == 0 and end == 0:
                token_idx[idx] = fill_val
            else:
                map_idx = char_idx[start:end]
                token_idx[idx] = statistics.mode(map_idx)
        token_indices.append(token_idx)
    return token_indices


def mapping_index_char2token_overlapped(
    char_indices: list[list[int]],
    input_ids: list[list[int]],
    offset_mappings: list[tuple[int, int]],
    overlap_mapping: list[int],
    fill_val: int = -1,
):
    """
    Map the index of the original token to the token position in the model, based on character position.
    This function is used for the overlapped tokenization.
    """
    token_indices = []
    for idx, ids, maps in zip(overlap_mapping, input_ids, offset_mappings):
        assert len(ids) == len(maps)
        char_idx = char_indices[idx]
        token_idx = np.zeros(len(ids), dtype=np.int32)
        for i, (start, end) in enumerate(maps):
            if start == 0 and end == 0:
                token_idx[i] = fill_val
            else:
                map_idx = char_idx[start:end]
                token_idx[i] = statistics.mode(map_idx)
        token_indices.append(token_idx)
    return token_indices


def get_char_pred_df(
    preds: np.ndarray,
    overlap_doc_ids: np.ndarray,
    offset_mappings: list[list[tuple[int, int]]],
    class_num: int,
):
    """
    文字単位に集約した予測値を取得する
    """
    doc_ids, char_indices, char_preds = [], [], []
    for pred, doc_id, offset in zip(preds, overlap_doc_ids, offset_mappings):
        for i in range(len(offset)):
            start, end = offset[i]
            if start == 0 and end == 0:
                continue

            doc_ids.extend([doc_id] * len(range(start, end)))
            char_indices.extend(list(range(start, end)))
            char_preds.extend([pred[i]] * len(range(start, end)))

    char_preds = np.stack(char_preds, axis=0)  # (char_len, class_num)
    char_pred_df = pl.DataFrame(char_preds, schema=[f"pred_{i}" for i in range(class_num)])
    char_pred_df = char_pred_df.with_columns(
        document=pl.Series(doc_ids).cast(pl.Int32),
        char_idx=pl.Series(char_indices).cast(pl.Int32),
    )
    char_pred_df = (
        char_pred_df.group_by("document", "char_idx")
        .agg([pl.col(f"pred_{i}").mean() for i in range(class_num)])
        .sort("document", "char_idx")
    )
    return char_pred_df


def get_char2org_df(
    doc_ids: list[int],
    full_text: list[str],
    org_tokens: list[list[str]],
    whitespaces: list[list[bool]],
):
    """
    文字とオリジナルトークンのマッピングをdfとして取得する
    """
    char2org_df = []
    for doc_id, text, org_token, whitespace in zip(doc_ids, full_text, org_tokens, whitespaces):
        char2org_map = np.zeros(len(text), dtype=np.int32)

        start_idx = 0
        for idx, (token, space) in enumerate(zip(org_token, whitespace)):
            char2org_map[start_idx : start_idx + len(token)] = idx
            start_idx += len(token)
            if space:
                char2org_map[start_idx] = -1
                start_idx += 1

        char2org_map = np.concatenate(
            [
                np.array([doc_id] * len(text)).reshape(-1, 1),
                np.array(list(range(len(text)))).reshape(-1, 1),
                char2org_map.reshape(-1, 1),
            ],
            axis=1,
        )
        char2org_df.append(char2org_map)

    char2org_df = np.concatenate(char2org_df, axis=0)
    char2org_df = pl.DataFrame(char2org_df, schema={"document": pl.Int32, "char_idx": pl.Int32, "token_idx": pl.Int32})
    return char2org_df


def get_pred_df(oof_df: pl.DataFrame, class_num: int, negative_th: float) -> pl.DataFrame:
    """
    全クラスの確率のテーブルから、最も確率の高いクラスを予測として取得する
    """
    pred_df = oof_df.with_columns(
        positive_pred=pl.Series(
            np.argmax(oof_df.select([f"pred_{i}" for i in range(1, class_num)]).to_numpy(), axis=1) + 1
        ),
        positive_prob=pl.Series(np.max(oof_df.select([f"pred_{i}" for i in range(1, class_num)]).to_numpy(), axis=1)),
        negative_prob=pl.col("pred_0"),
    )
    pred_df = pred_df.select(
        [
            pl.col("document"),
            pl.col("token_idx"),
            (
                pl.when(pl.col("negative_prob") > negative_th)
                .then(pl.col("negative_prob"))
                .otherwise(pl.col("positive_prob"))
                .alias("prob")
            ),
            (
                pl.when(pl.col("negative_prob") > negative_th)
                .then(pl.lit(0).cast(pl.Int8))
                .otherwise(pl.col("positive_pred").cast(pl.Int8))
                .alias("pred")
            ),
        ]
    )
    return pred_df.sort("document", "token_idx")


def get_pred_df_individual(
    oof_df: pl.DataFrame, class_num: int, negative_th_dict: dict[int, tuple[float, float]]
) -> pl.DataFrame:
    """
    全クラスの確率のテーブルから、最も確率の高いクラスを予測として取得する
    クラスごとに閾値を設定する
    """
    pred_df = oof_df.with_columns(
        positive_pred=pl.Series(
            np.argmax(oof_df.select([f"pred_{i}" for i in range(1, class_num)]).to_numpy(), axis=1) + 1
        ),
        positive_prob=pl.Series(np.max(oof_df.select([f"pred_{i}" for i in range(1, class_num)]).to_numpy(), axis=1)),
        negative_prob=pl.col("pred_0"),
    )
    pred_df_ = []
    mean_th = np.mean(list(filter(lambda x: x is not None, map(lambda x: x[0], negative_th_dict.values()))))
    for label_idx, label_df in pred_df.group_by("positive_pred"):
        negative_th, _ = negative_th_dict[label_idx]
        if negative_th is None:
            negative_th = mean_th

        label_df = label_df.select(
            [
                pl.col("document"),
                pl.col("token_idx"),
                (
                    pl.when(pl.col("negative_prob") > negative_th)
                    .then(pl.col("negative_prob"))
                    .otherwise(pl.col("positive_prob"))
                ).alias("prob"),
                (
                    pl.when(pl.col("negative_prob") > negative_th)
                    .then(pl.lit(0).cast(pl.Int8))
                    .otherwise(pl.col("positive_pred").cast(pl.Int8))
                ).alias("pred"),
            ]
        )
        pred_df_.append(label_df)

    pred_df = pl.concat(pred_df_).sort("document", "token_idx")
    return pred_df


def restore_prefix(config: DictConfig, pred_df: pl.DataFrame):
    """
    Restore Prefix, e.g. NAME_STUDENT -> B-NAME_STUDENT
    """
    # まずスペースに対しての予測を"O"に変更する -> これをしないと精度が著しく下がる
    org_token_df = get_original_token_df(config, pred_df["document"].unique().to_list())
    pred_df = pred_df.join(org_token_df, on=["document", "token_idx"], how="left", coalesce=True)
    pred_df = pred_df.with_columns(
        pred=(
            pl.when(pl.col("token").map_elements(lambda x: re.sub(r"[ \xa0]+", " ", x), return_dtype=pl.Utf8) == " ")
            .then(pl.lit(0))
            .otherwise(pl.col("pred"))
        )
    )
    pred_df = pred_df.drop(["token", "space"])

    # B, Iタグを割り当てる
    pred_df = pred_df.with_columns(org_pred=pl.col("pred").replace(IDX2TARGET_WTO_BIO, default="O"))
    pred_df = pred_df.with_columns(pred_diff=pl.col("pred").diff().over(["document"]).fill_null(-1))
    pred_df = pred_df.with_columns(
        prefix=(
            pl.when((pl.col("pred") != 0) & (pl.col("pred_diff") != 0))
            .then(pl.lit("B-"))
            .when((pl.col("pred") != 0) & (pl.col("pred_diff") == 0))
            .then(pl.lit("I-"))
            .otherwise(pl.lit(""))
        )
    )
    pred_df = pred_df.with_columns(
        pred=(
            pl.concat_str([pl.col("prefix"), pl.col("org_pred")])
            .replace(TARGET2IDX_WITH_BIO, default="0")
            .cast(pl.Int64)
        )
    )
    pred_df = pred_df.drop(["org_pred", "pred_diff", "prefix"])
    return pred_df


def get_original_token_df(config: DictConfig, document_ids: list[int]) -> pl.DataFrame:
    """
    オリジナルのデータに関するdfを取得する

    Returns:
        pl.DataFrame: [document, token_idx, token, space]
    """
    train_data = load_json_data(config.input_path / "train.json")
    org_token_df = []
    for data in train_data:
        doc_id = data["document"]
        org_tokens = data["tokens"]
        spaces = data["trailing_whitespace"]

        if doc_id in document_ids:
            org_token_df.append(
                pl.DataFrame(
                    dict(
                        document=pl.Series([doc_id] * len(org_tokens)).cast(pl.Int32),
                        token_idx=pl.Series(list(range(len(org_tokens)))).cast(pl.Int32),
                        token=org_tokens,
                        space=spaces,
                    ),
                )
            )
    org_token_df = pl.concat(org_token_df)
    return org_token_df


def get_truth_df(config: DictConfig, document_ids: list[int], convert_idx: bool) -> pl.DataFrame:
    """
    ラベルを元のトークンインデックスと共にデータフレームで取得

    Returns:
        pl.DataFrame: [document, token_idx, label]
    """
    train_data = load_json_data(config.input_path / "train.json")
    truth_df = []
    for data in train_data:
        doc_id = data["document"]
        labels = data["labels"]

        if doc_id in document_ids:
            truth_df.append(
                pl.DataFrame(
                    dict(
                        document=pl.Series([doc_id] * len(labels)).cast(pl.Int32),
                        token_idx=pl.Series(list(range(len(labels)))).cast(pl.Int32),
                        label=labels,
                    )
                )
            )

    truth_df = pl.concat(truth_df)
    if convert_idx:
        truth_df = truth_df.with_columns(label=pl.col("label").replace(TARGET2IDX_WITH_BIO, default=-1))
    return truth_df


def get_first_pred_df(
    config: DictConfig,
    oof_df: pl.DataFrame | None,
    oof_file_path: Path | None,
    document_ids: list[int] | None,
    negative_th: float,
):
    """
    1st stage(Detection)の予測確率からprefixありの予測結果を返す
    """
    assert oof_df is not None or oof_file_path is not None, "oof_df or oof_file_path must be given"
    if oof_df is None:
        oof_df = pl.read_parquet(oof_file_path)

    if document_ids is not None:
        oof_df = oof_df.filter(pl.col("document").is_in(document_ids))

    class_num = len(list(filter(lambda x: "pred" in x, oof_df.columns)))
    pred_df = get_pred_df(oof_df, class_num=class_num, negative_th=negative_th)
    if class_num == 8:
        pred_df = restore_prefix(config, pred_df)  # input_pathの情報が必要なのでconfigを渡す
    return pred_df
