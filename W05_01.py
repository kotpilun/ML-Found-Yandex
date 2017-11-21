import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

data = pd.read_csv('./DATA/W05_01.csv')

data.Sex = data.Sex.map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

y = data.Rings
X = data.loc[:, 'Sex':'ShellWeight']

cv = KFold(random_state=1, shuffle=True, n_splits=5)

scores = list()

for i in range(1,51):
    rnd_forest = RandomForestRegressor(random_state=1, n_estimators=i)
    rnd_forest.fit(X, y)
    scores.append(cross_val_score(rnd_forest, X, y, cv=cv, scoring='r2'))

scores_mean = pd.DataFrame(scores).mean(1)

for i in range(0,len(scores_mean)):
    if scores_mean[i] > 0.52:
        print(i+1)
        break

for i in range(0,len(scores_mean)):
    print(i+1, end=' - ')
    print(scores_mean[i])
# b = 0
# for i in range(0, len(scores)):
#     for j in scores[i]:
#         if j>0.52:
#             print(i)
#             b = 1
#             break
#     if b == 1:
#         break