# tested on Python 3.6.3
# work dir must contain: titanic.csv
# produces decision trees

import pandas        # http://pandas.pydata.org/
import numpy         # http://www.numpy.org/
import sklearn       # http://scikit-learn.org/stable/
import os            # https://docs.python.org/3/library/os.html
import graphviz      # https://pypi.python.org/pypi/graphviz

from sklearn.tree import DecisionTreeClassifier 

# set cd
os.chdir('D:\Programming\Python\IntroML\DecisionTree')

# load data from csv, only leave certain attributes and drop NA 
data = pandas.read_csv('titanic.csv', sep = '\t', index_col='PassengerId')
data = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']].copy()
data = data.dropna()

# convert entries for 'Sex' from string to 0,1 
data['Sex'] = (data['Sex'] !='female').astype(int)
X = numpy.array(data[['Pclass', 'Fare', 'Age', 'Sex']])

y = numpy.array(data[['Survived']])

# fit the tree 
clf    = DecisionTreeClassifier(random_state=241)
fitted = clf.fit(X, y)

# calculate the 2 most important factors
importances = fitted.feature_importances_
ar   = numpy.array([['Pclass', 'Fare', 'Age', 'Sex'],importances])
df   = pandas.DataFrame(ar.T, columns = ['Factors','Importance'])
df   = df.set_index('Factors')
sort = df.sort_values('Importance', ascending = False)
print("Two most important factors:\n", sort[0:2])

# visualize the tree 
dot_data = sklearn.tree.export_graphviz(fitted, out_file=None)
graph    = graphviz.Source(dot_data) 
graph