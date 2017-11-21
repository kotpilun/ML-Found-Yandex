import pandas as pd
from sklearn.decomposition import PCA
from numpy import corrcoef

data = pd.read_csv('./DATA/W04_02.csv')

pca =PCA(n_components=10)
pca.fit(data.loc[:,'AXP':])
# print(data)

s = 0
count = 0
for x in pca.explained_variance_ratio_:
    if s < 0.9:
        s += x
        count += 1

print(count, end=' ')
print('--', s)

pca_data = pd.DataFrame(pca.transform(data.loc[:,'AXP':]))
# print(pca_data[0])

data_dj = pd.read_csv('./DATA/W04_022.csv')
# print(data_dj)
corr = corrcoef(data_dj['^DJI'], pca_data[0])
print('correlation: ', round(corr[0,1],2))

company_weigth = pd.Series(pca.components_[0])
col = company_weigth.idxmax()
print(data.columns[col+1])
# print(data[company_weigth.idxmax()])