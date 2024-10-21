import re

import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm import tqdm

from src.utils.competition_utils import get_original_token_df, get_truth_df, load_json_data
from src.utils.constant import IDX2TARGET_WITH_BIO, TARGET2IDX_WITH_BIO


class PostProcessor:
    """
    Class for post-processing

    This class performs the following post-processing steps:
        - 1. Replace predictions made on whitespace characters with 0 ("O").
        - 2. Ensure the validity of prefix labels (B/I/O tags):
            - 2.1. If a "B" label is immediately followed by another "B" of the same type, change the subsequent "B" to "I".
            - 2.2. If an "I" label is preceded by an "O", and there is a "B" or "I" of the same type within a certain threshold distance, change the intervening "O"s to "I".
            - 2.3. If an "I" label is preceded by an "O" and condition 2-2 is not met, change the "I" to "B".
        - 3. Ensure the validity of label types:
            - 3.1. When multiple label types are mixed within a single entity, unify them into the most probable label type
            - 3.2. Remove false positives (FPs)
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
            truth_df = truth_df.with_columns(label_org=pl.col("label").replace(IDX2TARGET_WITH_BIO, default=""))
            pred_df = pred_df.join(truth_df, on=["document", "token_idx"], how="left", coalesce=True)

        pred_df = pred_df.with_columns(pred_org_prev=pl.col("pred_org"))

        # 1. Replace predictions on whitespace characters with 0 ("O").
        pred_df = self.remove_space_pii(pred_df)

        # 2. Ensure the validity of prefix labels (B/I/O tags).
        pred_df = self.check_prefix_validity(pred_df, rule1=True, rule2=False, rule3=True)
        # Reapply Step 1 because Step 2 may have introduced PII predictions on whitespace characters again.
        pred_df = self.remove_space_pii(pred_df)

        # 3. Ensure the validity of label types
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
        rule2: bool = False,
        rule3: bool = True,
        rule2_th: int = 2,
    ) -> pl.DataFrame:
        pred_df = pred_df.sort(["document", "token_idx"])

        pred_df_ = []
        for _, doc_df in tqdm(
            pred_df.group_by("document"),
            total=pred_df["document"].n_unique(),
            desc="Check Prefix Validity",
        ):
            preds_org = doc_df["pred_org"].to_numpy()
            new_preds_org = preds_org.copy()

            for i in range(len(doc_df)):
                pred_org = preds_org[i]
                if pred_org == "O":
                    continue

                prefix, label_type = pred_org.split("-")
                prefix_p1, label_type_p1 = self.get_prev_pred_label(new_preds_org, i - 1)

                # 2.1. Rule
                if rule1:
                    if prefix == "B" and prefix_p1 == "B" and label_type == label_type_p1:
                        new_preds_org[i] = f"I-{label_type}"

                if prefix == "I" and prefix_p1 == "O":
                    finding = False

                    # 2.2. Rule
                    if rule2:
                        for j in range(2, rule2_th + 1):
                            prefix_p, label_type_p = self.get_prev_pred_label(new_preds_org, i - j)
                            if prefix_p in ["B", "I"] and label_type == label_type_p:
                                finding = True
                                for k in range(1, j):
                                    new_preds_org[i - k] = f"I-{label_type}"

                    # 2.3. Rule
                    if rule3:
                        if not finding:
                            new_preds_org[i] = f"B-{label_type}"

                elif prefix == "I" and prefix_p1 is None:
                    new_preds_org[i] = f"B-{label_type}"

            doc_df = doc_df.with_columns(pred_org=pl.Series(new_preds_org.tolist()).cast(pl.Utf8)).with_columns(
                pred=pl.col("pred_org").replace(TARGET2IDX_WITH_BIO, default=0)
            )
            pred_df_.append(doc_df)

        pred_df = pl.concat(pred_df_)
        return pred_df

    def check_label_validity(self, pred_df: pl.DataFrame):
        # Assign a unique group ID to each PII.
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

        # Convert to pandas because Polars is extremely slow for this operation.
        # reason unknown; doesn't seem to be due to memory allocation or sorting issues.
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

            # Ensure that PII labels starts with "B" and the subsequent labels are all "I".
            for i, prefix in enumerate(group_df["prefix"].to_list()):
                if i == 0:
                    assert prefix == "B"
                else:
                    assert prefix == "I"

            group_df = group_df.with_columns(
                label_type=pl.col("pred_org").map_elements(lambda x: x.split("-")[1] if x != "O" else None)
            )

            # If label types differ within an PII groups, unify them to the label type with the highest cumulative probability.
            if group_df["label_type"].n_unique() > 1:
                highest_type = (
                    group_df.group_by("label_type")
                    .agg(pl.col("prob").sum())
                    .sort("prob", descending=True)["label_type"][0]
                )
                # Note that the probabilities of the changed rows remain unchanged
                group_df = group_df.with_columns(
                    pred_org=pl.concat_str([pl.col("prefix"), pl.lit("-"), pl.lit(highest_type)])
                )
                group_df = group_df.with_columns(pred=pl.col("pred_org").replace(TARGET2IDX_WITH_BIO, default=0))

            # Perform false positive (FP) removal based on specific criteria.
            group_df = self.remove_false_positive(group_df)
            pred_df_.append(group_df)

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
        phone_num: bool = True,
        url_personal: bool = False,
        street_address: bool = True,
        unq_count_rule: bool = False,
    ):
        label_type = pii_df["label_type"][0]
        pii_string = self.get_pii_string(pii_df)

        if name_student and label_type == "NAME_STUDENT":
            # Remove if the first character of the token is not uppercase and the token is a single character
            pii_df = pii_df.with_columns(
                pred=pl.when(pl.col("token").map_elements(lambda x: (x[0].isupper()) and (len(x.strip()) > 1)))
                .then(pl.col("pred"))
                .otherwise(pl.lit(0)),
                pred_org=pl.when(pl.col("token").map_elements(lambda x: (x[0].isupper()) and (len(x.strip()) > 1)))
                .then(pl.col("pred_org"))
                .otherwise(pl.lit("O")),
            )

        elif email and label_type == "EMAIL":
            # Remove if it does not match the following format -> [hoge]@[fuga].com|org|edu
            email_pattern = r"[^@ \t\r\n]+@[^@ \t\r\n]+\.(?:com|org|edu)"

            if re.search(email_pattern, pii_string) is None:
                pii_df = self.remove_pii(pii_df)
                return pii_df

        elif username and label_type == "USERNAME":
            # Remove tokens that are a single character
            pii_df = self.remove_pii_based_short_token(pii_df, len_th=1)

        elif id_num and label_type == "ID_NUM":
            # Remove tokens that are a single character
            pii_df = self.remove_pii_based_short_token(pii_df, len_th=1)

            # Remove if it consists only of digits and has fewer than N digits
            num_only = re.search(r"^\d+$", pii_string)
            if num_only is not None and len(pii_string) < 4:
                pii_df = self.remove_pii(pii_df)
                return pii_df

        elif phone_num and label_type == "PHONE_NUM":
            # Remove non-digit characters and remove if the number has fewer than N digits (minimum phone number length is 10 digits)
            num_string = re.sub(r"\D+", "", pii_string)
            if len(num_string) < 10:
                pii_df = self.remove_pii(pii_df)
                return pii_df

        elif url_personal and label_type == "URL_PERSONAL":
            # Remove if it does not match the following format -> # URLs starting with "http://" or "https://"
            url_pattern = r"^(?:http|https)://"
            if re.search(url_pattern, pii_string) is None:
                pii_df = self.remove_pii(pii_df)
                return pii_df

            # Remove if it contains the following
            remove_target = [
                "wikipedia.org",
            ]
            if any([target in pii_string for target in remove_target]):
                pii_df = self.remove_pii(pii_df)
                return pii_df

        elif street_address and label_type == "STREET_ADDRESS":
            # Remove if it is less than 10 characters
            if len(pii_string) < 10:
                pii_df = self.remove_pii(pii_df)
                return pii_df

        # Count the number of unique texts where pii_string appears -> Remove if it appears in texts of too many students
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
            .otherwise(pl.col("pred")),  # Remove tokens with a length less than or equal to char_len
            pred_org=pl.when(pl.col("token").map_elements(lambda x: len(x.strip()) <= len_th))
            .then(pl.lit("O"))
            .otherwise(pl.col("pred_org")),
        )
        return pii_df
