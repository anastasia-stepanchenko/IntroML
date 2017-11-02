# sosedi

import pandas as pd
import numpy

url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
data = pd.read_csv(url, header=None)
y = data.iloc[:,0]
X = data.iloc[:,1:14]

import sklearn
gen = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

def Accur(k):
    clf  = sklearn.neighbors.KNeighborsClassifier(n_neighbors = k)
    accur = sklearn.model_selection.cross_val_score(clf, X, y, cv=gen).mean()
    return accur

# raw attributes
Accuracy = list(map(Accur, range(51)[1:51]))
max(Accuracy)
print('Optimal number of neighbours with raw attributes =',\
      numpy.argmax(Accuracy)+1)


# scaled attributes
X = sklearn.preprocessing.scale(X)
Accuracy = list(map(Accur, range(51)[1:51]))
max(Accuracy)
print('Optimal number of neighbours with scaled attributes =',\
      numpy.argmax(Accuracy)+1)
