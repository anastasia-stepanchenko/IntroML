# tested on Python 3.6.3
# work dir must contain: data-logistic.csv
# performs logistic regressions with different penalties

import pandas       # http://pandas.pydata.org/
import sklearn      # http://scikit-learn.org/stable/
#import os           # https://docs.python.org/3/library/os.html

from sklearn.linear_model import LogisticRegression

# set cd
#os.chdir('D:\Programming\Python\IntroML\Logistic')

# load data from csv
data = pandas.read_csv('data-logistic.csv',  header=None)
X = data.iloc[:,1:3]
y = data.iloc[:,0]

# fit logistic regressions with different parameters of regularization
clf1 = LogisticRegression(penalty='l1')
clf2 = LogisticRegression(penalty='l2')
clf_noreg = LogisticRegression(C=10000)

clf1.fit(X,y)
clf2.fit(X,y)
clf_noreg.fit(X,y)

# prediction of all the models
prob1 = clf1.predict(X)
prob2 = clf1.predict(X)
prob_noreg = clf_noreg.predict(X)

# AUC-ROC scores for all models
auc1 = sklearn.metrics.roc_auc_score(y, prob1)
auc2 = sklearn.metrics.roc_auc_score(y, prob2)
auc_noreg = sklearn.metrics.roc_auc_score(y, prob_noreg)

print('AUC with regularization    =', round(auc1,4))
print('AUC without regularization =', round(auc_noreg,4))

