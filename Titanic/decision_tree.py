# decision trees

import os
cwd   = os.chdir('D:\Programming\Python\IntroML\Titanic')

import pandas as pd

data  = pd.read_csv('titanic.csv', sep = '\t', index_col='PassengerId')
data1 = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']].copy()
data1 = data1.dropna()
data1['Sex'] = (data1['Sex'] !='female').astype(int)

import numpy as np
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=241)

X = np.array(data1[['Pclass', 'Fare', 'Age', 'Sex']])
y = np.array(data1[['Survived']])
a = clf.fit(X, y)
importances = a.feature_importances_

ar = np.array([['Pclass', 'Fare', 'Age', 'Sex'],importances])
df = pd.DataFrame(ar.T, columns = ['Factors','Importance'])
df = df.set_index('Factors')
sort = df.sort_values('Importance', ascending = False)
print("Two most important factors:\n", sort[0:2])

import graphviz 
import sklearn as sk
dot_data = sk.tree.export_graphviz(a, out_file=None)
graph = graphviz.Source(dot_data) 
graph




# doesn't work
graph=pyd.graph_from_dot_file('k.dot')
graph.write_png("dtree.png")

dot_data = sk.tree.export_graphviz(a, out_file='k.dot') 
import pydotplus as pyd

graph = pyd.graph_from_dot_data(dot_data)

graph.render('D:\Programming\Python\Titanic\kk.dot')
graph.format = 'png'

from subprocess import check_call
check_call(['dot','-Tpng','D:\Programming\Python\Titanic\k.dot','-o','D:\Programming\Python\Titanic\k.png'])
