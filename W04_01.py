#Надо переписать под функции

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge                      #гребневая регрессия
import pandas as pd
from sklearn.feature_extraction import DictVectorizer       #кодирование категориальных признаков
from scipy.sparse import hstack

data_train = pd.read_csv('./DATA/W04_01.csv')


data_train['FullDescription'] = data_train['FullDescription'].map(lambda x: x.lower())
data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

print('Tfidfing FullDescription')
tfidf = TfidfVectorizer(min_df=5)
X_full_desc = tfidf.fit_transform(data_train['FullDescription'])

print('nan')
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

print('Vectorizing LocationNormalized and ContractTime')
enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))

print('hstack to X_train')
X_train = hstack([X_full_desc, X_train_categ])

print('Ridge Fitting')
regress = Ridge(alpha=1, random_state=241)
regress.fit(X_train, data_train.SalaryNormalized)

print('!!TEST DATA!!')
data_test = pd.read_csv('./DATA/W04_01_test.csv')



data_test['FullDescription'] = data_test['FullDescription'].map(lambda x: x.lower())
data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
#
# print('Tfidfing FullDescription')
X_full_desc_test = tfidf.transform(data_test['FullDescription'])

print('nan')
data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

print('Vectorizing LocationNormalized and ContractTime')
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

print('hstack to X_test')
X_test = hstack([X_full_desc_test, X_test_categ])

print('Ridge predicting!')
y_pred = regress.predict(X_test)

print(y_pred)
