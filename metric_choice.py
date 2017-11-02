# metric choice

import numpy
import sklearn.datasets
boston = sklearn.datasets.load_boston()
X = sklearn.preprocessing.scale(boston.data)
y = boston.target

gen = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=241)

def fMSE(par):
    clf  = sklearn.neighbors.KNeighborsRegressor(n_neighbors = 5, \
            metric = 'minkowski', p=par, weights='distance')
    mse = sklearn.model_selection.cross_val_score(clf, X, y, \
                    cv=gen,scoring='neg_mean_squared_error').mean()
    return mse


p = numpy.linspace(1,10,200)
MSE = list(map(fMSE, p))
max(MSE)
print('Optimal parameter of Mink.metric =', p[numpy.argmax(MSE)])
