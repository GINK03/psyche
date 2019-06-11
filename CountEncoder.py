
import pandas as pd
import numpy as np
from collections import Counter

def encode_with_columns_mutable(input_dfs=[], columns=[]):
    df = pd.concat(input_dfs)

    for column in columns:
        feat_freq = dict(Counter(df[column].tolist()))
        for adf in input_dfs:
            adf[f'{column}/CountEncode'] = adf[column].apply(lambda x:feat_freq[x])
        #print(column, feat_freq)
