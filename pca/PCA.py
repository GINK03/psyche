
from sklearn.decomposition import PCA
import pandas as pd
def append_pca_data(input_dfs=[], n_components=10):
    pca = PCA(n_components=n_components)
    same_col = set(input_dfs[0].columns) & set(input_dfs[1].columns)
    dfConc = pd.concat([input_dfs[0][same_col], input_dfs[1][same_col]], axis=0)
    #print(dfConc.shape)
    x = pca.fit_transform(dfConc.fillna(0).values)
    #print(x.shape) 
    dfTrain, dfTest = input_dfs
    size = len(dfTrain)
    for n in range(n_components):
        dfTrain[f'pca_{n:03d}'] = x[:,n][:size]
        dfTest[f'pca_{n:03d}'] = x[:,n][size:]
