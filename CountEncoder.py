
import pandas as pd
import numpy as np
from collections import Counter


def encode_with_columns_mutable(input_dfs=[], columns=[]):
    dfConc = pd.concat(input_dfs)

    for column in columns:
        feat_freq = dict(Counter(dfConc[column].fillna("None").tolist()))
        for df in input_dfs:
            df[f'CountEncode_{column}'] = df[column].fillna("None").apply(lambda x: feat_freq[x])
