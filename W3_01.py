import pandas as pn
from sklearn.svm import SVC

data = pn.read_csv('/Users/oleg/PycharmProjects/ML-Found-Yandex/DATA/w3_01.csv', header=None)
clf = SVC(kernel = 'linear', C = 100000, random_state = 241)

X = data.loc[:, 1:]
y = data.loc[:, 0]

# print(data)
clf.fit(X, y)

print(clf.support_)

# print(data)
