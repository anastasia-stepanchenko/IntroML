import pandas as pd
import os
cwd = os.chdir('D:\Programming\Python\IntroML\SVM')
#os.getcwd()

data = pd.read_csv('svm-data.csv', header=None)
X = data.iloc[:,1:3]
y = data.iloc[:,0]

import sklearn.svm
clf = sklearn.svm.SVC(C=100000, kernel='linear', random_state=241)
svm = clf.fit(X,y)
print('Support objects =', svm.support_)
