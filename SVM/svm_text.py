from sklearn import datasets
import numpy as np
import sklearn.svm
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )

X = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer()
td_matrix = vectorizer.fit_transform(X)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
gen = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=241)
clf = sklearn.svm.SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=gen)
gs.fit(td_matrix, y)

print('Best parameter :', gs.best_params_)

clf = sklearn.svm.SVC(C = 1, kernel='linear', random_state=241)
clf.fit(td_matrix, y)

k = np.array(clf.coef_.data)
heavyInd = k.argsort()[-10:]
heavyTerms = [q[i] for i in heavyInd]
print('10 words with largest weights:', heavyTerms[::-1])
