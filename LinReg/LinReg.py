# tested on Python 3.6.3
# work dir must contain: salary-train.csv, salary-test-mini.csv
# transforms texts to vectors and performs linear regression

import pandas     # http://pandas.pydata.org/
#import os         # https://docs.python.org/3/library/os.html

# http://scikit-learn.org/stable/modules/feature_extraction.html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge # http://scikit-learn.org/stable/
from scipy.sparse import hstack        # https://www.scipy.org/

# set cd
#os.chdir('D:\Programming\Python\IntroML\LinReg')

# load data from csv
train = pandas.read_csv('salary-train.csv')[0:10000]
test  = pandas.read_csv('salary-test-mini.csv')

# change capitalized letters
pandas.options.mode.chained_assignment = None
for i in range(len(train)):
    train['FullDescription'][i] = train['FullDescription'][i].lower()
for i in range(len(test)):
    test['FullDescription'][i] = test['FullDescription'][i].lower()

# leave only letters and numbers
train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]',\
     ' ', regex = True)
test['FullDescription']  = test['FullDescription'].replace('[^a-zA-Z0-9]',\
     ' ', regex = True)

# create term-doc matrix
vectorizer = TfidfVectorizer(min_df=5)
tdidf      = vectorizer.fit_transform(train.iloc[:,0])
tdidf_test = vectorizer.transform(test.iloc[:,0])

# fill empties with 'nan'
train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)
test['LocationNormalized'].fillna('nan', inplace=True)
test['ContractTime'].fillna('nan', inplace=True)

# categorical data to binary variables
enc = DictVectorizer()
X_train_categ = enc.fit_transform(train[['LocationNormalized', \
                                         'ContractTime']].to_dict('records'))
X_test_categ  = enc.transform(test[['LocationNormalized', \
                                   'ContractTime']].to_dict('records'))

# merge matrices
X_train = hstack([tdidf, X_train_categ])
X_test  = hstack([tdidf_test, X_test_categ])

# fit ridge regression and make predictions for test data
reg     = Ridge(alpha=1,random_state=241)
fitted  = reg.fit(X_train, train['SalaryNormalized'])
print('Predictions are', reg.predict(X_test))
