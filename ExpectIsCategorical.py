
import pandas as pd

def expect(input_dfs=[]):
    assert len(input_dfs) == 2, 'input_dfs size must be 2.'
    dfConcat = pd.concat(input_dfs)

    categories = []
    for column in dfConcat.columns:
        #print(column, str(dfConcat.dtypes))
        if str(dfConcat[column].dtypes) == 'object':
            #print(column, dfConcat[column].dtypes)
            categories.append(column)
    return categories
