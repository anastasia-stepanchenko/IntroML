# tested on Python 3.6.3
# transforms texts to vectors and performs SVM

import numpy        # http://www.numpy.org/
import sklearn.svm  # http://scikit-learn.org/stable/

from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

# load data from python datasets
newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )

X = newsgroups.data
y = newsgroups.target

# create term-doc matrix
vectorizer = TfidfVectorizer()
td_matrix  = vectorizer.fit_transform(X)

# find the best parameter for SVM via grid
grid = {'C': numpy.power(10.0, numpy.arange(-5, 6))}
gen  = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=241)
clf  = sklearn.svm.SVC(kernel='linear', random_state=241)
gs   = GridSearchCV(clf, grid, scoring='accuracy', cv=gen)
gs.fit(td_matrix, y)

print('Best parameter :', gs.best_params_)

# fit SVM
clf = sklearn.svm.SVC(C = 1, kernel='linear', random_state=241)
clf.fit(td_matrix, y)

# find 10 words with the highest weights
k = numpy.array(clf.coef_.data)
heavyInd   = k.argsort()[-10:]
q = vectorizer.get_feature_names() 
heavyTerms = [q[i] for i in heavyInd]
print('10 words with highest weights:', heavyTerms[::-1])
