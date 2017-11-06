# tested on Python 3.6.3
# defines the optimal parameter of Minkowski metrics for kNN regression 

import numpy            # http://www.numpy.org/
import sklearn.datasets # http://scikit-learn.org/stable/
import sklearn.neighbors
import sklearn.model_selection

# load data from sklearn.datasets
boston = sklearn.datasets.load_boston()
X      = sklearn.preprocessing.scale(boston.data)
y      = boston.target

# set up for cross-valid
gen = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=241)

# calculate MSE
def fMSE(par):
    clf = sklearn.neighbors.KNeighborsRegressor(n_neighbors = 5, \
            metric = 'minkowski', p=par, weights='distance')
    mse = sklearn.model_selection.cross_val_score(clf, X, y, \
                    cv=gen,scoring='neg_mean_squared_error').mean()
    return mse

# parameters to try
p = numpy.linspace(1,10,200)

# define the optimal parameter
MSE = list(map(fMSE, p))
max(MSE)
print('Optimal parameter of Mink.metric =', p[numpy.argmax(MSE)])
