from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pn


newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )
X = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)

score = 0
C = 0
for a in gs.grid_scores_:
    if a.mean_validation_score > score:
        score = a.mean_validation_score
        C = a.parameters['C']

clf = SVC(kernel='linear', random_state=241, C=C)
clf.fit(X, y)

words = vectorizer.get_feature_names()
coef = pn.DataFrame(clf.coef_.data, clf.coef_.indices)
# print('words:', words)
# print('coef', coef)
words = coef[0].map(lambda w: abs(w)).sort_values(ascending=False).head(10).index.map(lambda i: words[i])

print(words.sort_values())
     # print('оценка качества по кросс вылидации: ', a.mean_validation_score)
     # print('значения параметров: ', a.parameters)

# print(gs)

# print(grid)

# print(newsgroups.data)