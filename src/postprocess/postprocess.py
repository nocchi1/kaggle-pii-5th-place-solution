import re

import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm import tqdm

from src.utils.competition_utils import get_original_token_df, get_truth_df, load_json_data
from src.utils.constant import IDX2TARGET_WITH_BIO, TARGET2IDX_WITH_BIO


class PostProcessor:
    """
    後処理を行うクラス

    後処理一覧
        - 1. 空白文字の予測を0("O")に置き換える
        - 2. Prefixの妥当性を確保する
            - 2-1. B, Bが連続していて、かつ同一のタイプである場合に、後続のBをIに変更する
            - 2-2. Iの前がOで、かつ閾値以内に同一タイプのB, Iが存在する場合に、間のOをIに変更する
            - 2-3. Iの前がOで、かつ2-2の場合でない場合に、IをBに変更する
        - 3. ラベルのタイプが混在している場合に、それを1つのラベルに統合して、さらにFPを弾く処理を行う

    Expected DataFrame
        - pred_df (pl.DataFrame): [document, token_idx, prob, pred]
            - 全てのトークンに対しての予測を含める必要がある("O"の場合も除去しない)
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.org_data = load_json_data(config.input_path / f"{config.run_type}.json", debug=config.debug)
        self.full_texts = [d["full_text"] for d in self.org_data]

    def post_process(self, pred_df: pl.DataFrame) -> pl.DataFrame:
        pred_df = pred_df.with_columns(
            pred_org=pl.col("pred").replace(IDX2TARGET_WITH_BIO, default=""),
        )
        pred_doc_ids = pred_df["document"].unique().to_numpy()

        org_token_df = get_original_token_df(self.config, pred_doc_ids)
        pred_df = pred_df.join(org_token_df, on=["document", "token_idx"], how="left", coalesce=True)

        if self.config.run_type == "train":
            truth_df = get_truth_df(self.config, pred_doc_ids, convert_idx=True)
            truth_df = truth_df.with_columns(label_org=pl.col("label").repclace(IDX2TARGET_WITH_BIO, default=""))
            pred_df = pred_df.join(truth_df, on=["document", "token_idx"], how="left", coalesce=True)

        pred_df = pred_df.with_columns(pred_org_prev=pl.col("pred_org"))

        # 1. 空白文字の予測を0("O")に置き換える
        pred_df = self.remove_space_pii(pred_df)

        # 2. Prefixの妥当性を確保する
        pred_df = self.check_prefix_validity(pred_df, rule1=True, rule2=False, rule3=True)
        # 2の処理で空白に再度PIIが予測される場合があるため、再度1の処理を行う
        pred_df = self.remove_space_pii(pred_df)

        # 3. ラベルのタイプが混在している場合に、それを1つのラベルに統合して、さらにFPを弾く処理を行う
        pred_df = self.check_label_validity(pred_df)

        return pred_df

    def remove_space_pii(self, pred_df: pl.DataFrame) -> pl.DataFrame:
        pred_df = pred_df.with_columns(
            pred=(
                pl.when(pl.col("token").map_elements(lambda x: re.sub(r"[ \xa0]+", " ", x)) == " ")
                .then(pl.lit(0))
                .otherwise(pl.col("pred"))
            ),
            pred_org=(
                pl.when(pl.col("token").map_elements(lambda x: re.sub(r"[ \xa0]+", " ", x)) == " ")
                .then(pl.lit("O"))
                .otherwise(pl.col("pred_org"))
            ),
        )
        return pred_df

    def check_prefix_validity(
        self,
        pred_df: pl.DataFrame,
        rule1: bool = True,
        rule2: bool = True,
        rule3: bool = True,
        rule2_th: int = 2,
    ) -> pl.DataFrame:
        pred_df = pred_df.sort(["document", "token_idx"])

        pred_df_ = []
        for _, doc_df in tqdm(pred_df.group_by("document"), total=pred_df["document"].n_unique()):
            preds_org = doc_df["pred_org"].to_numpy()
            new_preds_org = preds_org.copy()

            for i in tqdm(range(len(doc_df)), desc="Check Prefix Validity"):
                pred_org = preds_org[i]
                if pred_org == "O":
                    continue

                prefix, label_type = pred_org.split("-")
                prefix_p1, label_type_p1 = self.get_prev_pred_label(new_preds_org, i - 1)

                # 1. B, Bが連続していて、かつ同一のタイプである場合に、後続のBをIに変更する
                if rule1:
                    if prefix == "B" and prefix_p1 == "B" and label_type == label_type_p1:
                        new_preds_org[i] = f"I-{label_type}"

                if prefix == "I" and prefix_p1 == "O":
                    finding = False

                    # 2. Iの前がOで、かつ閾値以内に同一タイプのB, Iが存在する場合に、間のOをIに変更する
                    if rule2:
                        for j in range(2, rule2_th + 1):
                            prefix_p, label_type_p = self.get_prev_pred_label(new_preds_org, i - j)
                            if prefix_p in ["B", "I"] and label_type == label_type_p:
                                finding = True
                                for k in range(1, j):
                                    new_preds_org[i - k] = f"I-{label_type}"

                    # 3. Iの前がOで、かつ2の場合でない場合に、IをBに変更する
                    if rule3:
                        if not finding:
                            new_preds_org[i] = f"B-{label_type}"

            doc_df = doc_df.with_columns(pred_org=pl.Series(new_preds_org.tolist()).cast(pl.Utf8)).with_columns(
                pred=pl.col("pred_org").replace(TARGET2IDX_WITH_BIO, default=0)
            )
            pred_df_.append(doc_df)

        pred_df = pl.concat(pred_df_)
        return pred_df

    def check_label_validity(self, pred_df: pl.DataFrame):
        # 各PIIに固有のグループIDを割り当てる
        pred_df = pred_df.with_row_index(name="group_idx").with_columns(
            prefix=pl.col("pred_org").map_elements(lambda x: x if x == "O" else x.split("-")[0])
        )
        pred_df = pred_df.with_columns(
            group_idx=(
                pl.when(pl.col("prefix") == "I")
                .then(pl.lit(None))
                .when(pl.col("prefix") == "O")
                .then(pl.lit(-1))
                .otherwise(pl.col("group_idx"))
            )
        )
        pred_df = pred_df.sort(["document", "token_idx"])
        pred_df = pred_df.with_columns(group_idx=pl.col("group_idx").fill_null(strategy="forward").over("document"))

        # 一度pandasに変換しないと激遅 (原因不明, メモリの配置やソートの問題ではないっぽい)
        pred_df = pred_df.to_pandas()
        pred_df = pl.from_pandas(pred_df)

        pred_df_ = []
        for (_, group_idx), group_df in tqdm(
            pred_df.group_by(["document", "group_idx"]),
            total=len(pred_df.unique(subset=["document", "group_idx"])),
            desc="Check PII Validity",
        ):
            if group_idx == -1:
                pred_df_.append(group_df)
                continue

            # Bから始まり、それ以外はIであることを確認
            for i, prefix in enumerate(group_df["prefix"].to_list()):
                if i == 0:
                    assert prefix == "B"
                else:
                    assert prefix == "I"

            group_df = group_df.with_columns(
                label_type=pl.col("pred_org").map_elements(lambda x: x.split("-")[1] if x != "O" else None)
            )

            # ラベルタイプが異なる場合は、最も確率が高いタイプに統一する
            if group_df["label_type"].n_unique() > 1:
                highest_type = (
                    group_df.group_by("label_type")
                    .agg(pl.col("prob").sum())
                    .sort("prob", descending=True)["label_type"][0]
                )
                # 変更される行の確率は元のままであることに注意 -> Ensembleの手法はVotingだからあまり関係はない
                group_df = group_df.with_columns(
                    pred_org=pl.concat_str([pl.col("prefix"), pl.lit("-"), pl.lit(highest_type)])
                )
                group_df = group_df.with_columns(pred=pl.col("pred_org").replace(TARGET2IDX_WITH_BIO, default=0))

            # FPを弾く処理を行う
            group_df = self.remove_false_positive(group_df)
            pred_df_.append(group_df)

        # 全体を結合する
        pred_df_ = (
            pl.concat(pred_df_, how="diagonal")
            .sort("document", "token_idx")
            .drop(["group_idx", "prefix", "label_type"])
        )
        return pred_df_

    def get_prev_pred_label(self, preds_array: np.ndarray, idx: int):
        if idx >= 0:
            label = preds_array[idx]
            if label == "O":
                return "O", None
            else:
                return label.split("-")
        else:
            return None, None

    def remove_false_positive(
        self,
        pii_df: pl.DataFrame,
        name_student: bool = True,
        email: bool = True,
        username: bool = True,
        id_num: bool = True,
        phone_num: bool = False,
        url_personal: bool = False,
        street_address: bool = False,
        unq_count_rule: bool = False,
    ):
        label_type = pii_df["label_type"][0]
        pii_string = self.get_pii_string(pii_df)

        if name_student and label_type == "NAME_STUDENT":
            # トークンの先頭の文字が大文字でない場合かつ,1文字の場合に除去
            pii_df = pii_df.with_columns(
                pred=pl.when(pl.col("token").map_elements(lambda x: (x[0].isupper()) and (len(x.strip()) > 1)))
                .then(pl.col("pred"))
                .otherwise(pl.lit(0)),
                pred_org=pl.when(pl.col("token").map_elements(lambda x: (x[0].isupper()) and (len(x.strip()) > 1)))
                .then(pl.col("pred_org"))
                .otherwise(pl.lit("O")),
            )

        elif email and label_type == "EMAIL":
            # 下記のフォーマットでマッチしない場合は除去する
            email_pattern = r"[^@ \t\r\n]+@[^@ \t\r\n]+\.(?:com|org|edu)"  # [任意の文字列]@[任意の文字列].com|org|edu
            if re.search(email_pattern, pii_string) is None:
                pii_df = self.remove_pii(pii_df)
                return pii_df

        elif username and label_type == "USERNAME":
            # トークンが1文字の場合に除去
            pii_df = self.remove_pii_based_short_token(pii_df, len_th=1)

        elif id_num and label_type == "ID_NUM":
            # トークンが1文字の場合に除去
            pii_df = self.remove_pii_based_short_token(pii_df, len_th=1)

            # N桁未満の数字のみの場合に除去
            num_only = re.search(r"^\d+$", pii_string)
            if num_only is not None and len(pii_string) < 4:
                pii_df = self.remove_pii(pii_df)
                return pii_df

        elif phone_num and label_type == "PHONE_NUM":
            # 数字以外を除去して数字がN桁(電話番号の最小桁数は10桁)未満の場合は除去する
            num_string = re.sub(r"\D+", "", pii_string)
            if len(num_string) < 10:
                pii_df = self.remove_pii(pii_df)
                return pii_df

        elif url_personal and label_type == "URL_PERSONAL":
            # 下記のフォーマットでマッチしない場合は除去する
            url_pattern = r"^(?:http|https)://"  # http://, https://で始まるURL
            if re.search(url_pattern, pii_string) is None:
                pii_df = self.remove_pii(pii_df)
                return pii_df

            # 以下を含む場合は除去する
            remove_target = [
                "wikipedia.org",
            ]
            if any([target in pii_string for target in remove_target]):
                pii_df = self.remove_pii(pii_df)
                return pii_df

        elif street_address and label_type == "STREET_ADDRESS":
            # 10文字未満であれば除去する
            if len(pii_string) < 10:
                pii_df = self.remove_pii(pii_df)
                return pii_df

        # pii_stringが出現するテキストのユニークな数をカウントする -> 一定以上の生徒のテキストで出現する場合は除去する
        if unq_count_rule:
            unq_appear_count = len([text for text in self.full_texts if pii_string in text])
            condition = ((unq_appear_count >= 3) and (label_type != "NAME_STUDENT")) or (
                (unq_appear_count >= 10) and (label_type == "NAME_STUDENT")
            )
            if condition:
                pii_df = self.remove_pii(pii_df)

        return pii_df

    def get_pii_string(self, pii_df: pl.DataFrame):
        pii_string = ""
        for token, space in pii_df[["token", "space"]].to_numpy():
            if space:
                pii_string += token + " "
            else:
                pii_string += token
        return pii_string.strip()

    def remove_pii(self, pii_df: pl.DataFrame):
        pii_df = pii_df.with_columns(pred=pl.lit(0).cast(pl.Int64), pred_org=pl.lit("O"))
        return pii_df

    def remove_pii_based_short_token(self, pii_df: pl.DataFrame, len_th: int):
        pii_df = pii_df.with_columns(
            pred=pl.when(pl.col("token").map_elements(lambda x: len(x.strip()) <= len_th))
            .then(pl.lit(0))
            .otherwise(pl.col("pred")),  # char_len以下の文字数のトークンを除去
            pred_org=pl.when(pl.col("token").map_elements(lambda x: len(x.strip()) <= len_th))
            .then(pl.lit("O"))
            .otherwise(pl.col("pred_org")),
        )
        return pii_df
