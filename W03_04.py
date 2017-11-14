import  pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

data = pd.read_csv('./DATA/classification.csv')

TP, FP, TN, FN = 0, 0, 0, 0

for i in range(0,len(data)):
    if data.true[i] == 1 and data.pred[i] == 1:
        TP += 1
    elif data.true[i] == 0 and data.pred[i] == 1:
        FP += 1
    elif data.true[i] == 0 and data.pred[i] == 0:
        TN += 1
    elif data.true[i] == 1 and data.pred[i] == 0:
        FN += 1

print(TP, FP, FN, TN)
print('accuracy score: ', round(accuracy_score(data.true, data.pred), 2))
print('precision score: ', round(precision_score(data.true, data.pred), 2))
print('recall score: ', round(recall_score(data.true, data.pred), 2))
print('f1_score: ', round(f1_score(data.true, data.pred), 2))

print('---------------------')

scores = pd.read_csv('./DATA/scores.csv')
print('score_logreg: ', round(roc_auc_score(scores.true, scores.score_logreg), 2))
print('score svm: ', round(roc_auc_score(scores.true, scores.score_svm), 2))
print('score_knn: ', round(roc_auc_score(scores.true, scores.score_knn), 2))
print('score_tree: ', round(roc_auc_score(scores.true, scores.score_tree), 2))

print('---------------------')


#good decision from github
pres_rec_scores = {}
for i in scores.columns[1:]: #получить наименования колонок
    recall_curve = precision_recall_curve(scores.true, scores[i]) #scores[i] - данные по наименованию колонки
    pres_rec = pd.DataFrame({'precision': recall_curve[0], 'recall': recall_curve[1]})
    pres_rec_scores[i] = pres_rec[pres_rec['recall'] >= 0.7]['precision'].max() #фильтация по заданному порогу, поиск макс точности

print(pd.Series(pres_rec_scores).sort_values(ascending=False)) #трансформация из словаря в Series, первый столбец ключи, второй значения