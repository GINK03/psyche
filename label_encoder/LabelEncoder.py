from sklearn.preprocessing import LabelEncoder as SkLabelEncoder
import pandas as pd

def encode_with_columns_mutable(input_dfs=[], columns=[], fillna=True):
    assert len(input_dfs) == 2, "need train and test dataframes"
    for column in columns:
        if fillna is False:
            assert not input_dfs[0][column].isnull().values.any(), f'[CRIT] there is nullable in {column}'

    for column in columns:
        le = SkLabelEncoder()
        input = pd.concat([input_dfs[0][column], input_dfs[1][column]])
        if fillna:
            input = input.fillna('___NULL___')
            le.fit(input)
            input_dfs[0][column] = le.transform(input_dfs[0][column].fillna('___NULL___'))
            input_dfs[1][column] = le.transform(input_dfs[1][column].fillna('___NULL___'))
        else:
            le.fit(input)
            input_dfs[0][column] = le.transform(input_dfs[0][column])
            input_dfs[1][column] = le.transform(input_dfs[1][column])
