import re
import polars as pl
from tqdm import tqdm
import numpy as np

from src.utils.constant import IDX2TARGET_WITH_BIO, TARGET2IDX_WITH_BIO
from src.data.load import load_competition_data
from src.data.utils import get_original_token_df, get_truth_df


class PostProcessor:
    """
    pred_df(各トークンに対しての予測ラベルを決定した後のDataFrame)に対して、後処理を行うクラス
    pred_dfは全てのトークンに対しての予測を含める("O"の場合も除去しない)
    e.g. pred_df
    document | token_index | prob | pred
    ------------------------------------
    1        | 0           | 0.9  | 1
    1        | 1           | 0.8  | 8
    1        | 2           | 0.9  | 0
    """
    
    def __init__(self, config, pred_df: pl.DataFrame):
        self.config = config
        self.pred_df = pred_df.with_columns(
            pred_org = pl.col('pred').replace(IDX2TARGET_WITH_BIO, default='')
        )
        self.pred_doc_ids = sorted(self.pred_df['document'].unique().to_list())
        all_data = load_competition_data(config, config.run_type)
        self.full_text_list = [d['full_text'] for d in all_data]
        
        org_token_df = get_original_token_df(config, self.pred_doc_ids)
        self.pred_df = self.pred_df.join(org_token_df, on=['document', 'token_index'], how='left')
        if config.run_type == 'train':
            truth_df = get_truth_df(config, self.pred_doc_ids, is_label_idx=False)
            truth_df = truth_df.with_columns(label_org = pl.col('label').replace(IDX2TARGET_WITH_BIO, default=''))
            self.pred_df = self.pred_df.join(truth_df, on=['document', 'token_index'], how='left')
    
        
    def post_process(self, pred_df: pl.DataFrame) -> pl.DataFrame:
        # 変更箇所がわかりやすいように、元の予測を残しておく
        pred_df = pred_df.with_columns(
            prev_pred_org = pl.col('pred_org')
        )
        # 後処理を行う
        pred_df = self.remove_space_pii(pred_df) # 1. 空白文字に対してPIIを予測している場合に、そのトークンの予測を0("O")に置き換える
        pred_df = self.check_bio_validity(pred_df) # 2. ラベルの妥当性を確保する
        pred_df = self.remove_space_pii(pred_df) # 2の処理で空白に再度PIIが予測される場合があるため、再度1の処理を行う
        pred_df = self.check_pii_validity(pred_df) # 3. ラベルのタイプが混在している場合に、それを1つのラベルに統合して、さらにFPを弾く処理を行う
        return pred_df
    
    
    def get_post_processed_df(self):
        pred_df = self.post_process(self.pred_df)
        return pred_df

        
    def remove_space_pii(self, pred_df: pl.DataFrame):
        pred_df = pred_df.with_columns(
            pred = (
                pl.when(pl.col('token').map_elements(lambda x: re.sub(r'[ \xa0]+', ' ', x)) == ' ')
                .then(pl.lit(0))
                .otherwise(pl.col('pred'))
            ),
            pred_org = (
                pl.when(pl.col('token').map_elements(lambda x: re.sub(r'[ \xa0]+', ' ', x)) == ' ')
                .then(pl.lit('O'))
                .otherwise(pl.col('pred_org'))
            )
        )
        return pred_df
    
    
    def get_prev_pred_label(self, preds_array: np.ndarray, idx: int):
        if idx >= 0:
            label = preds_array[idx]
            if label == 'O':
                return 'O', None
            else:
                return label.split('-')
        else:
            return None, None
    
    
    def check_bio_validity(self, pred_df: pl.DataFrame, continue_th: int=1):
        """
        1. B, Bが連続していて、かつ同一のタイプである場合に、後続のBをIに変更する
        2. Iの前がOで、かつ閾値以内に同一タイプのB, Iが存在する場合に、間のOをIに変更する
        3. Iの前がOで、かつ2の場合でない場合に、IをBに変更する
        """
        pred_pp = []
        for _, doc_df in tqdm(pred_df.group_by('document'), total=pred_df['document'].n_unique()):
            doc_df = doc_df.sort('token_index')
            preds_org = doc_df['pred_org'].to_numpy()
            new_preds_org = preds_org.copy() # 後処理後の予測ラベル
            
            for i in range(len(doc_df)):
                if preds_org[i] == 'O': # Oの場合はスキップ
                    continue
                
                prefix_c, label_type_c = preds_org[i].split('-')
                prefix_p1, label_type_p1 = self.get_prev_pred_label(preds_org, i-1) # 1つ前のラベルを取得
                
                # 1. B, Bが連続していて、かつ同一のタイプである場合に、後続のBをIに変更する
                if prefix_c == 'B' and prefix_p1 == 'B' and label_type_c == label_type_p1:
                    new_preds_org[i] = f'I-{label_type_c}'
                
                # 2. Iの前がOで、かつ閾値以内に同一タイプのB, Iが存在する場合に、間のOをIに変更する
                if prefix_c == 'I' and prefix_p1 not in ['B', 'I']:
                    # 閾値以内に同一タイプのB, Iが存在するかどうか
                    find_flag = False
                    # for j in range(2, continue_th + 2):
                    #     prefix_p, label_type_p = self.get_prev_pred_label(new_preds_org, i-j)
                    #     if prefix_p in ['B', 'I'] and label_type_c == label_type_p:
                    #         find_flag = True
                    #         for k in range(1, j):
                    #             new_preds_org[i-k] = f'I-{label_type_c}' # 閾値以内に同一タイプのB, Iが存在する場合はOをIに変更
                                
                    # 3. Iの前がOで、かつ2の場合でない場合に、IをBに変更する
                    if find_flag == False:
                        new_preds_org[i] = f'B-{label_type_c}'
                
            doc_df = (
                doc_df.with_columns(pred_org = pl.Series(new_preds_org.tolist()).cast(pl.Utf8))
                .with_columns(pred = pl.col('pred_org').replace(TARGET2IDX_WITH_BIO, default=0))
            )
            pred_pp.append(doc_df)
            
        pred_pp = pl.concat(pred_pp)
        return pred_pp
    
    
    def get_pii_string(self, pii_df: pl.DataFrame):
        pii_string = ''
        for token, space in pii_df[['token', 'space']].to_numpy():
            if space:
                pii_string += token + ' '
            else:
                pii_string += token
        return pii_string.strip()
    
    
    def remove_token_base(self, pii_df: pl.DataFrame, char_len: int):
        pii_df = pii_df.with_columns(
            pred = pl.when(pl.col('token').map_elements(lambda x: len(x.strip()) <= char_len)).then(pl.lit(0)).otherwise(pl.col('pred')), # char_len以下の文字数のトークンを除去
            pred_org = pl.when(pl.col('token').map_elements(lambda x: len(x.strip()) <= char_len)).then(pl.lit('O')).otherwise(pl.col('pred_org')),
        )
        return pii_df
        
        
    def remove_pii_base(self, pii_df: pl.DataFrame):
        pii_df = (
            pii_df.with_columns(
                pred = pl.lit(0).cast(pl.Int64),
                pred_org = pl.lit('O')
            )
        )
        return pii_df


    def remove_false_positive(self, pii_df: pl.DataFrame):
        pii_type = pii_df['pii_type'][0]
        pii_string = self.get_pii_string(pii_df)
        
        if pii_type == 'NAME_STUDENT':
            # トークンの先頭の文字が大文字でない場合かつ,1文字の場合に除去
            pii_df = pii_df.with_columns(
                pred = pl.when(pl.col('token').map_elements(lambda x: (x[0].isupper()) and (len(x.strip()) > 1))).then(pl.col('pred')).otherwise(pl.lit(0)),
                pred_org = pl.when(pl.col('token').map_elements(lambda x: (x[0].isupper()) and (len(x.strip()) > 1))).then(pl.col('pred_org')).otherwise(pl.lit('O'))
            )
            
        elif pii_type == 'EMAIL':
            # 下記のフォーマットでマッチしない場合は除去する
            email_pattern = r'[^@ \t\r\n]+@[^@ \t\r\n]+\.(?:com|org|edu)' # [任意の文字列]@[任意の文字列].com|org|edu
            if re.search(email_pattern, pii_string) is None:
                pii_df = self.remove_pii_base(pii_df)
                return pii_df
        
        elif pii_type == 'USERNAME':
            # トークンが1文字の場合に除去
            pii_df = self.remove_token_base(pii_df, char_len=1)
            
        elif pii_type == 'ID_NUM':
            # トークンが1文字の場合に除去
            pii_df = self.remove_token_base(pii_df, char_len=1)
            # N桁未満の数字のみの場合に除去
            num_only = re.search(r'^\d+$', pii_string)
            if num_only is not None and len(pii_string) < 4: # 4桁未満の数字のみの場合に除去
                pii_df = self.remove_pii_base(pii_df)
                return pii_df
            
        elif pii_type == 'PHONE_NUM':
            # 数字以外を除去して数字がN桁(電話番号の最小桁数は10桁)未満の場合は除去する -> 全体を取りきれていないケースも存在しそうなのでFNを取り切るようにした方がいいかも
            num_string = re.sub(r'\D+', '', pii_string)
            if len(num_string) < 10:
                pii_df = self.remove_pii_base(pii_df)
                return pii_df
        
        # elif pii_type == 'URL_PERSONAL':
        #     # # 下記のフォーマットでマッチしない場合は除去する -> 他のprefixも追加する
        #     # url_pattern = r'^(?:http|https)://' # http://, https://で始まるURL
        #     # if re.search(url_pattern, pii_string) is None:
        #     #     pii_df = self.remove_pii_base(pii_df)
        #     #     return pii_df
            
        #     # 以下を含む場合は除去する
        #     remove_target = [
        #         'wikipedia.org', 
        #     ]
        #     if any([target in pii_string for target in remove_target]):
        #         pii_df = self.remove_pii_base(pii_df)
        #         return pii_df
        
        elif pii_type == 'STREET_ADDRESS':
            # 10文字未満であれば除去する -> もっと桁数を増やしてもいいかもしれない
            if len(pii_string) < 10:
                pii_df = self.remove_pii_base(pii_df)
                return pii_df
            
        # # pii_stringが出現するテキストのユニークな数をカウントする -> 一定以上の生徒のテキストで出現する場合は除去する
        # unique_appear_count = sum([1 for text in self.full_text_list if pii_string in text])
        # condition = (
        #     ((unique_appear_count >= 2) and (pii_type != 'NAME_STUDENT'))
        #     # ((unique_appear_count >= 4) and (pii_type == 'NAME_STUDENT') and len(pii_df) >= 2)
        # )
        # if condition:
        #     pii_df = self.remove_pii_base(pii_df)
        return pii_df
    
    def check_pii_validity(self, pred_df: pl.DataFrame):
        # 各PIIに固有のグループIDを割り当てる
        pred_df = (
            pred_df.with_row_index(name='pii_group')
            .with_columns(
                pii_prefix = pl.col('pred_org').map_elements(lambda x: x if x == 'O' else x.split('-')[0])
            )
        )
        pred_df = pred_df.with_columns(
            pii_group = (
                pl.when(pl.col('pii_prefix') == 'I').then(pl.lit(None))
                .when(pl.col('pii_prefix') == 'O').then(pl.lit(-1))
                .otherwise(pl.col('pii_group'))
            )
        )
        pred_df = pred_df.sort(['document', 'token_index'])
        pred_df = pred_df.with_columns(
            pii_group = pl.col('pii_group').fill_null(strategy='forward').over('document')
        )
        
        # 一度pandasに変換して再度polarsに変換を行う -> これを行わないと激遅 (原因不明, メモリの配置やソートの問題ではないっぽい)
        pred_df = pred_df.to_pandas()
        pred_df = pl.from_pandas(pred_df)
        
        pred_pp = []
        for (_, pii_group), pii_df in tqdm(
            pred_df.group_by(['document', 'pii_group']), 
            total=len(pred_df.unique(subset=['document', 'pii_group']))
        ):
            if pii_group == -1:
                pred_pp.append(pii_df)
                continue
            
            # Bから始まり、それ以外はIであることを確認
            for i, prefix in enumerate(pii_df['pii_prefix'].to_list()):
                if i == 0:
                    assert prefix == 'B'
                else:
                    assert prefix == 'I'

            pii_df = pii_df.with_columns(
                pii_type = pl.col('pred_org').map_elements(lambda x: x.split('-')[1] if x != 'O' else None)
            )
            
            # PIIタイプが異なる場合は、最も確率が高いタイプに統一する
            if pii_df['pii_type'].n_unique() > 1:
                highest_type = pii_df.group_by('pii_type').agg(pl.col('prob').sum()).sort('prob', descending=True)['pii_type'][0]
                pii_df = pii_df.with_columns(
                    pred_org = pl.concat_str([
                        pl.col('pii_prefix'),
                        pl.lit('-'),
                        pl.lit(highest_type)
                    ])
                )
                pii_df = pii_df.with_columns(
                    pred = pl.col('pred_org').replace(TARGET2IDX_WITH_BIO, default=0)
                )
                
            # # FPを弾く処理を行う
            pii_df = self.remove_false_positive(pii_df)
            pred_pp.append(pii_df)
            
        # 全体を結合する
        pred_pp = (
            pl.concat(pred_pp, how='diagonal')
            .sort('document', 'token_index')
            .drop(['pii_group', 'pii_prefix', 'pii_type'])
        )
        return pred_pp
    
    
# class PostProcessor:
#     """
#     pred_df(各トークンに対しての予測ラベルを決定した後のDataFrame)に対して、後処理を行うクラス
#     pred_dfは全てのトークンに対しての予測を含める("O"の場合も除去しない)
#     e.g. pred_df
#     document | token_index | prob | pred
#     ------------------------------------
#     1        | 0           | 0.9  | 1
#     1        | 1           | 0.8  | 8
#     1        | 2           | 0.9  | 0
#     """
    
#     def __init__(self, config, pred_df: pl.DataFrame):
#         self.config = config
#         self.pred_df = pred_df.with_columns(
#             pred_org = pl.col('pred').replace(IDX2TARGET_WITH_BIO, default='')
#         )
#         self.pred_doc_ids = sorted(self.pred_df['document'].unique().to_list())
#         all_data = load_competition_data(config, config.run_type)
#         self.full_text_list = [d['full_text'] for d in all_data]
        
#         org_token_df = get_original_token_df(config, self.pred_doc_ids)
#         self.pred_df = self.pred_df.join(org_token_df, on=['document', 'token_index'], how='left')
#         if config.run_type == 'train':
#             truth_df = get_truth_df(config, self.pred_doc_ids, is_label_idx=True)
#             truth_df = truth_df.with_columns(label_org = pl.col('label').replace(IDX2TARGET_WITH_BIO, default=''))
#             self.pred_df = self.pred_df.join(truth_df, on=['document', 'token_index'], how='left')
    
        
#     def post_process(self, pred_df: pl.DataFrame) -> pl.DataFrame:
#         # 変更箇所がわかりやすいように、元の予測を残しておく
#         pred_df = pred_df.with_columns(
#             prev_pred_org = pl.col('pred_org')
#         )
#         # 後処理を行う
#         pred_df = self.remove_space_pii(pred_df) # 1. 空白文字に対してPIIを予測している場合に、そのトークンの予測を0("O")に置き換える
#         pred_df = self.check_bio_validity(pred_df) # 2. ラベルの妥当性を確保する
#         pred_df = self.remove_space_pii(pred_df) # 2の処理で空白に再度PIIが予測される場合があるため、再度1の処理を行う
#         pred_df = self.check_pii_validity(pred_df) # 3. ラベルのタイプが混在している場合に、それを1つのラベルに統合して、さらにFPを弾く処理を行う
#         return pred_df
    
    
#     def get_post_processed_df(self):
#         pred_df = self.post_process(self.pred_df)
#         return pred_df

        
#     def remove_space_pii(self, pred_df: pl.DataFrame):
#         pred_df = pred_df.with_columns(
#             pred = (
#                 pl.when(pl.col('token').map_elements(lambda x: re.sub(r'[ \xa0]+', ' ', x)) == ' ')
#                 .then(pl.lit(0))
#                 .otherwise(pl.col('pred'))
#             ),
#             pred_org = (
#                 pl.when(pl.col('token').map_elements(lambda x: re.sub(r'[ \xa0]+', ' ', x)) == ' ')
#                 .then(pl.lit('O'))
#                 .otherwise(pl.col('pred_org'))
#             )
#         )
#         return pred_df
    
    
#     def get_prev_pred_label(self, preds_array: np.ndarray, idx: int):
#         if idx >= 0:
#             label = preds_array[idx]
#             if label == 'O':
#                 return 'O', None
#             else:
#                 return label.split('-')
#         else:
#             return None, None
    
    
#     def check_bio_validity(self, pred_df: pl.DataFrame, continue_th: int=1):
#         """
#         1. B, Bが連続していて、かつ同一のタイプである場合に、後続のBをIに変更する
#         2. Iの前がOで、かつ閾値以内に同一タイプのB, Iが存在する場合に、間のOをIに変更する
#         3. Iの前がOで、かつ2の場合でない場合に、IをBに変更する
#         """
#         pred_pp = []
#         for _, doc_df in tqdm(pred_df.group_by('document'), total=pred_df['document'].n_unique()):
#             doc_df = doc_df.sort('token_index')
#             preds_org = doc_df['pred_org'].to_numpy()
#             new_preds_org = preds_org.copy() # 後処理後の予測ラベル
            
#             for i in range(len(doc_df)):
#                 if preds_org[i] == 'O': # Oの場合はスキップ
#                     continue
                
#                 prefix_c, label_type_c = preds_org[i].split('-')
#                 prefix_p1, label_type_p1 = self.get_prev_pred_label(preds_org, i-1) # 1つ前のラベルを取得
                
#                 # 1. B, Bが連続していて、かつ同一のタイプである場合に、後続のBをIに変更する -> これよくないかも
#                 if prefix_c == 'B' and prefix_p1 == 'B' and label_type_c == label_type_p1:
#                     new_preds_org[i] = f'I-{label_type_c}'
                
#                 # 2. Iの前がOで、かつ閾値以内に同一タイプのB, Iが存在する場合に、間のOをIに変更する
#                 if prefix_c == 'I' and prefix_p1 not in ['B', 'I']:
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
#                         new_preds_org[i] = f'B-{label_type_c}'
                
#             doc_df = (
#                 doc_df.with_columns(pred_org = pl.Series(new_preds_org.tolist()).cast(pl.Utf8))
#                 .with_columns(pred = pl.col('pred_org').replace(TARGET2IDX_WITH_BIO, default=0))
#             )
#             pred_pp.append(doc_df)
            
#         pred_pp = pl.concat(pred_pp)
#         return pred_pp
    
    
#     def get_pii_string(self, pii_df: pl.DataFrame):
#         pii_string = ''
#         for token, space in pii_df[['token', 'space']].to_numpy():
#             if space:
#                 pii_string += token + ' '
#             else:
#                 pii_string += token
#         return pii_string.strip()
    
    
#     def remove_token_base(self, pii_df: pl.DataFrame, char_len: int):
#         pii_df = pii_df.with_columns(
#             pred = pl.when(pl.col('token').map_elements(lambda x: len(x.strip()) <= char_len)).then(pl.lit(0)).otherwise(pl.col('pred')), # char_len以下の文字数のトークンを除去
#             pred_org = pl.when(pl.col('token').map_elements(lambda x: len(x.strip()) <= char_len)).then(pl.lit('O')).otherwise(pl.col('pred_org')),
#         )
#         return pii_df
        
        
#     def remove_pii_base(self, pii_df: pl.DataFrame):
#         pii_df = (
#             pii_df.with_columns(
#                 pred = pl.lit(0).cast(pl.Int64),
#                 pred_org = pl.lit('O')
#             )
#         )
#         return pii_df


#     def remove_false_positive(self, pii_df: pl.DataFrame):
#         pii_type = pii_df['pii_type'][0]
#         pii_string = self.get_pii_string(pii_df)
        
#         if pii_type == 'NAME_STUDENT':
#             # トークンの先頭の文字が大文字でない場合かつ,1文字の場合に除去
#             pii_df = pii_df.with_columns(
#                 pred = pl.when(pl.col('token').map_elements(lambda x: (x[0].isupper()) and (len(x.strip()) > 1))).then(pl.col('pred')).otherwise(pl.lit(0)),
#                 pred_org = pl.when(pl.col('token').map_elements(lambda x: (x[0].isupper()) and (len(x.strip()) > 1))).then(pl.col('pred_org')).otherwise(pl.lit('O'))
#             )
            
#         elif pii_type == 'EMAIL':
#             # 下記のフォーマットでマッチしない場合は除去する
#             email_pattern = r'[^@ \t\r\n]+@[^@ \t\r\n]+\.(?:com|org|edu)' # [任意の文字列]@[任意の文字列].com|org|edu
#             if re.search(email_pattern, pii_string) is None:
#                 pii_df = self.remove_pii_base(pii_df)
#                 return pii_df
        
#         elif pii_type == 'USERNAME':
#             # 各トークンが2文字以下の場合に除去
#             pii_df = self.remove_token_base(pii_df, char_len=2)
            
#         elif pii_type == 'ID_NUM':
#             # 各トークンが2文字以下の場合に除去
#             pii_df = self.remove_token_base(pii_df, char_len=3)
#             # N桁未満の数字のみの場合に除去
#             num_only = re.search(r'^\d+$', pii_string)
#             if num_only is not None and len(pii_string) < 4: # 4桁未満の数字のみの場合に除去
#                 pii_df = self.remove_pii_base(pii_df)
#                 return pii_df
            
#         elif pii_type == 'PHONE_NUM':
#             # 数字以外を除去して数字がN桁(電話番号の最小桁数は10桁)未満の場合は除去する -> 全体を取りきれていないケースも存在しそうなのでFNを取り切るようにした方がいいかも
#             num_string = re.sub(r'\D+', '', pii_string)
#             if len(num_string) < 10:
#                 pii_df = self.remove_pii_base(pii_df)
#                 return pii_df
        
#         elif pii_type == 'URL_PERSONAL':
#             # 下記のフォーマットでマッチしない場合は除去する -> 他のprefixも追加する
#             url_pattern = r'^(?:http|https|ftp|mailto|file|tel|sftp|git|magnet|ws|wss|rtsp|sip|sips|jdbc|ldap|ldaps|nntp|irc|gopher|telnet|ttp|tp|ttps|tps)'
#             if re.search(url_pattern, pii_string) is None:
#                 pii_df = self.remove_pii_base(pii_df)
#                 return pii_df
            
#             # 以下を含む場合は除去する
#             # remove_target = [
#             #     'wikipedia.org', 
#             # ]
#             # if any([target in pii_string for target in remove_target]):
#             #     pii_df = self.remove_pii_base(pii_df)
#             #     return pii_df
        
#         elif pii_type == 'STREET_ADDRESS':
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
#         pred_df = (
#             pred_df.with_row_index(name='pii_group')
#             .with_columns(
#                 pii_prefix = pl.col('pred_org').map_elements(lambda x: x if x == 'O' else x.split('-')[0])
#             )
#         )
#         pred_df = pred_df.with_columns(
#             pii_group = (
#                 pl.when(pl.col('pii_prefix') == 'I').then(pl.lit(None))
#                 .when(pl.col('pii_prefix') == 'O').then(pl.lit(-1))
#                 .otherwise(pl.col('pii_group'))
#             )
#         )
#         pred_df = pred_df.sort(['document', 'token_index'])
#         pred_df = pred_df.with_columns(
#             pii_group = pl.col('pii_group').fill_null(strategy='forward').over('document')
#         )
        
#         # 一度pandasに変換して再度polarsに変換を行う -> これを行わないと激遅 (原因不明, メモリの配置やソートの問題ではないっぽい)
#         pred_df = pred_df.to_pandas()
#         pred_df = pl.from_pandas(pred_df)
        
#         pred_pp = []
#         for (_, pii_group), pii_df in tqdm(
#             pred_df.group_by(['document', 'pii_group']), 
#             total=len(pred_df.unique(subset=['document', 'pii_group']))
#         ):
#             if pii_group == -1:
#                 pred_pp.append(pii_df)
#                 continue
            
#             # Bから始まり、それ以外はIであることを確認
#             for i, prefix in enumerate(pii_df['pii_prefix'].to_list()):
#                 if i == 0:
#                     assert prefix == 'B'
#                 else:
#                     assert prefix == 'I'

#             pii_df = pii_df.with_columns(
#                 pii_type = pl.col('pred_org').map_elements(lambda x: x.split('-')[1] if x != 'O' else None)
#             )
            
#             # PIIタイプが異なる場合は、最も確率が高いタイプに統一する
#             if pii_df['pii_type'].n_unique() > 1:
#                 highest_type = pii_df.group_by('pii_type').agg(pl.col('prob').sum()).sort('prob', descending=True)['pii_type'][0]
#                 pii_df = pii_df.with_columns(
#                     pred_org = pl.concat_str([
#                         pl.col('pii_prefix'),
#                         pl.lit('-'),
#                         pl.lit(highest_type)
#                     ])
#                 )
#                 pii_df = pii_df.with_columns(
#                     pred = pl.col('pred_org').replace(TARGET2IDX_WITH_BIO, default=0)
#                 )
                
#             # # FPを弾く処理を行う
#             pii_df = self.remove_false_positive(pii_df)
#             pred_pp.append(pii_df)
            
#         # 全体を結合する
#         pred_pp = (
#             pl.concat(pred_pp, how='diagonal')
#             .sort('document', 'token_index')
#             .drop(['pii_group', 'pii_prefix', 'pii_type'])
#         )
#         return pred_pp