# tested on Python 3.6.3
# work dir must contain: close_prices.csv, djia_index.csv
# performs PCA and gets insights about components

import pandas     # http://pandas.pydata.org/
import numpy      # http://www.numpy.org/
#import os         # https://docs.python.org/3/library/os.html

from sklearn.decomposition import PCA # http://scikit-learn.org/stable/

# set cd
#os.chdir('D:\Programming\Python\IntroML\PCA')

# load data from csv
data = pandas.read_csv('close_prices.csv').iloc[:, 1:32]
dj   = pandas.read_csv('djia_index.csv').iloc[:,1]

# fit PCA with 10 components
pca = PCA(10)
pca.fit(data)

# find a company with the highest weight in the 1st component
maxWeightInd = pca.components_[0].argmax()
print('Company with the highest impact to the 1st component is', \
      list(data)[maxWeightInd])

# define sufficient number of components for 90% of explained varience
totRat = 0
for i in range(len(pca.explained_variance_ratio_)):
    totRat += pca.explained_variance_ratio_[i]
    if totRat>=0.90:
        print('Sufficient number of components for 90% e.v. =',i+1)
        break

# apply the model to data 
result = pca.transform(data)

# calculate Pearson corr between the 1st component and DJ index
comp1 = result[:,0]
print('Pearson corr between the 1st component and DJ index =', \
      round(numpy.corrcoef(comp1,dj)[0,1],2))
