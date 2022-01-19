import pandas as pd
import re
from typing import List, Tuple


def filter_spam_tweets(x: pd.DataFrame):
    """
    スパムの可能性があるツイートを削除
    Args:
        - x
            - x["text"], x["source"]にアクセスできるデータフレーム
    Returns:
        - pd.DataFrame:
            - フィルタを行ったデータフレーム
    """
    x = x[pd.notna(x["text"]) & pd.notna(x["source"])]
    x.query('source.str.contains("(Android|iPhone)", regex=True)',
            inplace=True)
    x.query('not text.str.contains("#")', inplace=True)
    x.query('not text.str.contains("http")', inplace=True)
    x.query('not text.str.contains("^RT", regex=True)', inplace=True)
    return x


def get_morphemes(parser, x: str) -> List[Tuple]:
    """
    ツイートを形態素解析して必要な情報をTupleで返す
    Args:
        - x
            - ツイート
    Returns:
        - List[Tuple]
            - 形態素解析した結果
    """
    data = []
    for line in parser.parse(x).strip().split("\n"):
        entities = line.split("\t")
        if len(entities) <= 4:
            continue
        word = entities[0]  # もとの単語
        yomi = entities[1]  # 読み
        orig = entities[2]  # 未活用の原型
        if len(orig) == 1:
            continue
        mtype = entities[3]  # 品詞
        # 形容詞, 動詞 -> wordをorigへ
        if re.search("(^形容詞|^動詞)", mtype):
            elm = (orig, yomi, mtype)
        else:
            elm = (word, yomi, mtype)
        data.append(elm)
    return data

def filter_noise_words(col_name:str, x: pd.DataFrame) -> pd.DataFrame:
    res = x.query(f'not {col_name}.str.contains("(「|」【|】|『|』\(|\^|\)|\*|`|（|）|~|〜|\!|\.|┈|#|！)", regex=True)')
    return res
    


