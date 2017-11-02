import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
import sklearn

cwd = os.chdir('D:\Programming\Python\IntroML\Logistic')
#os.getcwd()

data = pd.read_csv('data-logistic.csv',  header=None)
X = data.iloc[:,1:3]
y = data.iloc[:,0]

clf1 = LogisticRegression(penalty='l1')
clf2 = LogisticRegression(penalty='l2')
clf_noreg = LogisticRegression(C=10000)

clf1.fit(X,y)
clf2.fit(X,y)
clf_noreg.fit(X,y)

prob1 = clf1.predict(X)
prob2 = clf1.predict(X)
prob_noreg = clf_noreg.predict(X)

auc1 = sklearn.metrics.roc_auc_score(y, prob1)
auc2 = sklearn.metrics.roc_auc_score(y, prob2)
auc_noreg = sklearn.metrics.roc_auc_score(y, prob_noreg)

print('AUC with regularization =', round(auc1,2))
