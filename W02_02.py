from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pn

data = load_boston()
scale(data.data)
cv = KFold(n_splits=5,shuffle=True,random_state=42)

acc = list()

for p in np.linspace(start=1,stop=10,num=200):
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance',p=p)
    acc.append(cross_val_score(neigh, data.data, data.target,cv=cv,scoring='neg_mean_squared_error').mean())

# print(acc) -21.056657
print(pn.DataFrame({'acc':acc, 'p':np.linspace(start=1,stop=10,num=200)}).sort_values(by=['acc'],ascending=False))