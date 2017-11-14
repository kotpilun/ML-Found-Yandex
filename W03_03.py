import pandas as pn
import math
from sklearn.metrics import roc_auc_score

data = pn.read_csv('./DATA/W03_03.csv', header=None)

X = data.loc[:, 1:]
y = data.loc[:, 0]

S1, S2, w1, w2, w1_past, w2_past = 0,0,0,0,0,0
j = 0

#un-regularized
# while j <= 10000:
#     w1_grad, w1_past = w1, w1
#     w2_grad, w2_past = w2, w2
#     S1 = 0
#     S2 = 0
#     for i in range(0, len(y)):
#         S1 += y[i] * X[1][i] * (1-1/(1+math.exp(-y[i] * (w1_grad * X[1][i] + w2_grad * X[2][i]))))
#
#     for i in range(0, len(y)):
#         S2 += y[i] * X[2][i] * (1-1/(1+math.exp(-y[i] * (w1_grad * X[1][i] + w2_grad * X[2][i]))))
#
#     w1 = w1 + (0.1 * 1/len(y) * S1)
#     w2 = w2 + (0.1 * 1/len(y) * S2)
#
#     if math.sqrt((w1_past - w1) ** 2 + (w2_past - w2) ** 2) <= 0.00001:
#         break
#     j += 1

#regularized
while j <= 10000:
    w1_grad, w1_past = w1, w1
    w2_grad, w2_past = w2, w2
    S1 = 0
    S2 = 0
    for i in range(0, len(y)):
        S1 += y[i] * X[1][i] * (1-1/(1+math.exp(-y[i] * (w1_grad * X[1][i] + w2_grad * X[2][i]))))

    for i in range(0, len(y)):
        S2 += y[i] * X[2][i] * (1-1/(1+math.exp(-y[i] * (w1_grad * X[1][i] + w2_grad * X[2][i]))))

    w1 = w1 + (0.1 * 1/len(y) * S1) - 10 * 10 * w1_grad
    w2 = w2 + (0.1 * 1/len(y) * S2) - 10 * 10 * w2_grad

    if math.sqrt((w1_past - w1) ** 2 + (w2_past - w2) ** 2) <= 0.00001:
        break
    j += 1

ax = pn.Series()
ax =[1 / (1 + math.exp(-w1*X[1][i] - w2*X[2][i])) for i in range(0,len(y))]

print(roc_auc_score(y, ax))