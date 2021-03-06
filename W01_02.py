import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pn

data = pn.read_csv('./DATA/titanic.csv', index_col='PassengerId')

data_for_tree = pn.DataFrame(data, columns=['Pclass','Fare','Age','Sex','Survived']).dropna()
# data.dropna()

surv = pn.DataFrame(data_for_tree, columns=['Survived'])

del data_for_tree['Survived']

data_for_tree.Sex = data_for_tree.Sex.map(lambda x: 1 if x == 'male' else 0)
# data_for_tree.Sex = [1 for i in data_for_tree.Sex if i == 'male']

#преобразуем строку в числа: male = 1, female = 0
# for i in data_for_tree.index:
#     if data_for_tree.get_value(index=i,col='Sex') == 'male':
#         data_for_tree.set_value(index=i,col='Sex', value=1)
#     else:
#         data_for_tree.set_value(index=i,col='Sex', value=0)
#
#
# clf = DecisionTreeClassifier(random_state=241)
# clf = clf.fit(data_for_tree,surv)
#
# imp = clf.feature_importances_

print(data_for_tree.Age)












































































































