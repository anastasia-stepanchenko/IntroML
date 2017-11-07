# tested on Python 3.6.3
# work dir must contain: gbm-data.csv
# performs gradient boosting with decisionn trees, defines optimal number
# of iterations and visualises logloss plot

import pandas        # http://pandas.pydata.org/
import sklearn       # http://scikit-learn.org/stable/
import os            # https://docs.python.org/3/library/os.html
import numpy         # http://www.numpy.org/
import math          # https://docs.python.org/2/library/math.html
import matplotlib.pyplot as plt  # https://matplotlib.org/api/pyplot_api.html

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# set cd
os.chdir('D:\Programming\Python\IntroML\GradBoost')
# load data from csv
data = pandas.read_csv('gbm-data.csv').values
X = data[:,1:1777]
y = data[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, \
                                      random_state=42)

# choose lr (learning rate) out of [1, 0.5, 0.3, 0.2, 0.1] 
lr = 0.2
print('Learning rate =', lr)

# fit gradient boosting
clf = GradientBoostingClassifier(n_estimators=250, verbose=True, \
                                     learning_rate = lr, random_state=241)
clf.fit(X_train, y_train)
    
# retrieve predictions on each iteration
stage_train  = list(clf.staged_decision_function(X_train))
stage_test   = list(clf.staged_decision_function(X_test))

# convert predictions to the probability range
for i in range(len(stage_train)):
    for j in range(len(stage_train[0])):
        stage_train[i][j] = 1 / (1 + math.exp(- stage_train[i][j]))
for i in range(len(stage_test)):
    for j in range(len(stage_test[0])):
        stage_test[i][j]  = 1 / (1 + math.exp(- stage_test[i][j]))

# calculate logloss on each iteration
logloss_train  = [sklearn.metrics.log_loss(y_train, stage_train[i]) \
                     for i in range(len(stage_train))]
logloss_test   = [sklearn.metrics.log_loss(y_test, stage_test[i]) \
                     for i in range(len(stage_test))]    

# define the optimal number of iteration
optnum = numpy.argmin(logloss_test)+1
print('optimal number of iterations =', optnum)
print('min of logloss               =', min(logloss_test))

# visualize logloss
plt.figure()
plt.plot(logloss_test, 'r', logloss_train, 'g', linewidth=2)
plt.legend(['test', 'train'])
plt.show()

# fit random forest for these data and # of trees = optim.num of iterations
clf = RandomForestClassifier(n_estimators = optnum, random_state=241)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

# compare with logloss of random forest
logloss_rf   = sklearn.metrics.log_loss(y_test, y_pred)    
print('logloss of random forest     =', logloss_rf)


