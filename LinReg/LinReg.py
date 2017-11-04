import pandas as pd
import sklearn
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

os.chdir('D:\Programming\Python\IntroML\LinReg')
raw_train = pd.read_csv('salary-train.csv')[0:10000]
test  = pd.read_csv('salary-test-mini.csv')

train = raw_train.copy()

for i in range(len(train)):
    train['FullDescription'][i] = train['FullDescription'][i].lower()
for i in range(2):
    test['FullDescription'][i] = test['FullDescription'][i].lower()

train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]',\
     ' ', regex = True)
test['FullDescription']  = test['FullDescription'].replace('[^a-zA-Z0-9]',\
     ' ', regex = True)

vectorizer = TfidfVectorizer(min_df=5)
tdidf = vectorizer.fit_transform(train.iloc[:,0])
tdidf_test = vectorizer.transform(test.iloc[:,0])

train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)
test['LocationNormalized'].fillna('nan', inplace=True)
test['ContractTime'].fillna('nan', inplace=True)


enc = DictVectorizer()
X_train_categ = enc.fit_transform(train[['LocationNormalized', \
                                         'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(test[['LocationNormalized', \
                                   'ContractTime']].to_dict('records'))

X_train = hstack([tdidf, X_train_categ])
X_test  = hstack([tdidf_test, X_test_categ])

reg = Ridge(alpha=1,random_state=241)
fitted = reg.fit(X_train, train['SalaryNormalized'])
print('Predictions are', reg.predict(X_test))
