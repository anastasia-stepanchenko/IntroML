# tested on Python 3.6.3
# work dir must contain: svm-data.csv
# performs SVM and prints the support vestors

import pandas       # http://pandas.pydata.org/
import sklearn.svm  # http://scikit-learn.org/stable/
import os           # https://docs.python.org/3/library/os.html

# set cd
os.chdir('D:\Programming\Python\IntroML\SVM')

# load data from csv
data = pandas.read_csv('svm-data.csv', header=None)
X    = data.iloc[:,1:3]
y    = data.iloc[:,0]

# fit SVM
clf  = sklearn.svm.SVC(C=100000, kernel='linear', random_state=241)
svm  = clf.fit(X,y)
print('Support objects =', svm.support_)