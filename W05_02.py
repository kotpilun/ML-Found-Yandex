import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import math
import matplotlib.pyplot as plt

# %matplotlib inline

data = pd.read_csv('./DATA/W05_02.csv')
X = data.loc[:, 'D1':'D1776']
y = data['Activity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)


# for i in [1, 0.5, 0.3, 0.2, 0.1]:
#     acc_train = list()
#     acc_test = list()
#     clf = GradientBoostingClassifier(n_estimators=250, verbose=False, random_state=241, learning_rate=i)
#     clf.fit(X_train, y_train)
#
#     for x in clf.staged_decision_function(X_train):
#         acc_train.append([1.0 / (1.0 + math.exp(-i)) for i in x])
#     acc_train = list(map(lambda x: log_loss(y_train, x), acc_train))
#
#     for x in clf.staged_decision_function(X_test):
#         acc_test.append([1.0 / (1.0 + math.exp(-i)) for i in x])
#     acc_test = list(map(lambda x: log_loss(y_test, x), acc_test))

    # plt.figure()
    # plt.plot(acc_test, 'r', linewidth=2)
    # plt.plot(acc_train, 'g', linewidth=2)
    # plt.legend(['test', 'train'])
    # plt.show()

acc_train = list()
acc_test = list()
clf = GradientBoostingClassifier(n_estimators=250, verbose=False, random_state=241, learning_rate=0.2)
clf.fit(X_train, y_train)

for x in clf.staged_decision_function(X_test):
    acc_test.append([1.0 / (1.0 + math.exp(-i)) for i in x])
acc_test = list(map(lambda x: log_loss(y_test, x), acc_test))
acc_test = pd.Series(acc_test)
print(acc_test.sort_values(ascending=True).head(1))

rf_clf = RandomForestClassifier(n_estimators=36, random_state=241)
rf_clf.fit(X_train, y_train)
pred_y = rf_clf.predict_proba(X_test)[:,1]
loss = log_loss(y_test, pred_y)

print(loss)