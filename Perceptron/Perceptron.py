from IPython import get_ipython
get_ipython().magic('reset -sf')

import pandas as pd
import os
os.chdir('D:\Programming\Python\IntroML\Perceptron')
#os.getcwd()

datafit = pd.read_csv('perceptron-train.csv', header=None)
datatest = pd.read_csv('perceptron-test.csv', header=None)

X = datafit.iloc[:,1:3]
y = datafit.iloc[:,0]
Xtest = datatest.iloc[:,1:3]
ytest = datatest.iloc[:,0]

import sklearn
from sklearn.linear_model import Perceptron
clf = Perceptron(random_state=241)
#predictions = clf.predict(X)

gen = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
accur = sklearn.model_selection.cross_val_score(clf, Xtest, ytest, cv=gen).mean()
print('Accuracy before normalization =', accur)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Xtest_scaled = scaler.transform(Xtest)

clf.fit(X_scaled, y)
accur_scaled = sklearn.model_selection.cross_val_score(clf, Xtest_scaled, \
                                                       ytest, cv=gen).mean()
print('Accuracy after normalization  =', accur_scaled)

