import pandas as pn
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data_train = pn.read_csv('/Users/oleg/PycharmProjects/ML-Found-Yandex/DATA/train.csv', header = None)
data_test = pn.read_csv('/Users/oleg/PycharmProjects/ML-Found-Yandex/DATA/test.csv', header = None)

clf = Perceptron(random_state=241)
clf.fit(data_train.loc[:, 1:], data_train.loc[:, 0])
predictions = clf.predict(data_test.loc[:, 1:])
accuracy_before = accuracy_score(data_test.loc[:, 0], predictions)
# print('accuracy before normalize: ', accuracy)

#Normalize data sets
scaler = StandardScaler()
data_train_scaled = scaler.fit_transform(data_train.loc[:, 1:])
data_test_scaled = scaler.transform(data_test.loc[:, 1:])

y_train_scaled = data_train.loc[:, 0]
y_test_scaled = data_test.loc[:, 0]

clf.fit(data_train_scaled, y_train_scaled)
predictions = clf.predict(data_test_scaled)
accuracy_after = accuracy_score(y_test_scaled, predictions)
print(accuracy_after - accuracy_before)
print(accuracy_after)
# print(data_test)
