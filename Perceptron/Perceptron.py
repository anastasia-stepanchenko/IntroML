## tested on Python 3.6.3
# work dir must contain: perceptron-train.csv, perceptron-test.csv
# performs perceptron method and compares accuracies with and without 
# normalization of data

import pandas     # http://pandas.pydata.org/
import sklearn    # http://scikit-learn.org/stable/
#import os         # https://docs.python.org/3/library/os.html

from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

# set cd
#os.chdir('D:\Programming\Python\IntroML\Perceptron')

# load data from csv
datafit  = pandas.read_csv('perceptron-train.csv', header=None)
datatest = pandas.read_csv('perceptron-test.csv', header=None)

X     = datafit.iloc[:,1:3]
y     = datafit.iloc[:,0]
Xtest = datatest.iloc[:,1:3]
ytest = datatest.iloc[:,0]

# fit and calculate accuracy of perceptron model
clf    = Perceptron(random_state=241, max_iter = 1000)
fitted = clf.fit(X, y)
accur  = sklearn.metrics.accuracy_score(ytest, clf.predict(Xtest))
print('Accuracy before normalization =', accur)

# normalize data
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
Xtest_scaled = scaler.transform(Xtest)

# fit with normalized data and calculate accuracy
fitted2      = clf.fit(X_scaled, y)
accur_scaled = sklearn.metrics.accuracy_score(ytest, clf.predict(Xtest_scaled))
print('Accuracy after normalization  =', accur_scaled)