# tested on Python 3.6.3
# needs Internet access
# defines the optimal number of neighbors in kNN method

import pandas                  # http://pandas.pydata.org/
import numpy                   # http://www.numpy.org/
import sklearn.model_selection # http://scikit-learn.org/stable/

from sklearn.neighbors import KNeighborsClassifier

# load data from url (takes some minutes)
url  ="https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
data = pandas.read_csv(url, header=None)
y    = data.iloc[:,0]
X    = data.iloc[:,1:14]

# set up cross-validation
gen = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

# function that calculates accuracy of kNN method 
# depanding on the number of neighbous
def Accur(k):
    clf   = KNeighborsClassifier(n_neighbors = k)
    accur = sklearn.model_selection.cross_val_score(clf, X, y, cv=gen).mean()
    return accur

# calculate accuracy and define the optimal k for raw values of attributes
Accuracy = list(map(Accur, range(51)[1:51]))
max(Accuracy)
print('Optimal number of neighbours with raw attributes =',\
      numpy.argmax(Accuracy)+1)

# scale the attributes, calculate accuracy and define the optimal k for scaled
# values of attributes
X        = sklearn.preprocessing.scale(X)
Accuracy = list(map(Accur, range(51)[1:51]))
max(Accuracy)
print('Optimal number of neighbours with scaled attributes =',\
      numpy.argmax(Accuracy)+1)