
import matplotlib.pyplot as plt
import seaborn as sns
def get(df):
    fig, ax = plt.subplots(figsize=(15,15))
    # calculate the correlation matrix
    corr = df[:100000].corr()

    # plot the heatmap
    return sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns, ax=ax)
