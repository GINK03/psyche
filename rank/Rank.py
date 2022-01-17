import pandas as pd


def rank(input_dfs=[], col=None):
    assert isinstance(col, str), 'col must be str object'
    dfTrain, dfTest = input_dfs
    size = len(dfTrain)
    dfConc = pd.concat([dfTrain[[col]], dfTest[[col]]], axis=0)
    dfConc[f'Rank_{col}'] = dfConc[c].rank()
    dfTrain[f'Rank_{col}'] = dfConc[f'Rank_{col}'][:size]
    dfTest[f'Rank_{col}'] = dfConc[f'Rank_{col}'][size:]


def ranks(input_dfs=[], cols=None):
    assert isinstance(cols, list), 'cols must be list object'
    dfTrain, dfTest = input_dfs
    size = len(dfTrain)

    for col in cols:
        dfConc = pd.concat([dfTrain[[col]], dfTest[[col]]], axis=0)
        dfConc[f'Rank_{col}'] = dfConc[col].rank()
        dfTrain[f'Rank_{col}'] = dfConc[f'Rank_{col}'][:size]
        dfTest[f'Rank_{col}'] = dfConc[f'Rank_{col}'][size:]
