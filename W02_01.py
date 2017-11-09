import pandas as pn
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

data = pn.read_csv('/home/oleg/PycharmProjects/ML-Found-Yandex/Data/Wine2.data')
y = data.res
del data['res']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

#accuracy = {}
acc = list()

for i in range(1,51):
    neigh = KNeighborsClassifier(i)
    acc.append(cross_val_score(neigh, data, y, cv=kf).mean())

# print(pn.Series(acc).sort_values(ascending=False))
# Normalize data
acc = list()
data = preprocessing.scale(data)
for i in range(1,51):
    neigh = KNeighborsClassifier(i)
    acc.append(cross_val_score(neigh, data, y, cv=kf).mean())

# print(pn.Series(acc))
print(pn.Series(acc).sort_values(ascending=False))