## tested on Python 3.6.3
# work dir must contain: abalone.csv
# performs random forest method and defines optimal number of trees

import pandas        # http://pandas.pydata.org/
import sklearn       # http://scikit-learn.org/stable/
import os            # https://docs.python.org/3/library/os.html

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer

# set cd
os.chdir('D:\Programming\Python\IntroML\RandomForest')

# load data from csv, only leave certain attributes and drop NA 
data = pandas.read_csv('abalone.csv')

# convert 'Sex' from string to  1/-1/0
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = data.iloc[:,0:len(list(data))-1]
y = data.iloc[:,len(list(data))-1]

# set up cross-validation
gen = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=1)

# calculates r2 quality metric
def r2(k):
    reg = RandomForestRegressor(n_estimators = k, random_state=1)
    r2sc  = cross_val_score(reg, X, y, cv = gen,\
            scoring = make_scorer(sklearn.metrics.r2_score)).mean()
    return r2sc
    
result = [r2(k+1) for k in range(50)]

# define minimal number of trees to reach quality 0.52 
for i in range(len(result)):
    if result[i] >0.52: 
        minInd = i
        break
    
print('Minimal number of trees to reach quality 0.52 =', minInd+1)
