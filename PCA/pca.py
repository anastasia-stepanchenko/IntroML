import pandas as pd
import numpy
import os
from sklearn.decomposition import PCA

os.chdir('D:\Programming\Python\IntroML\PCA')
data = pd.read_csv('close_prices.csv').iloc[:, 1:32]
dj   = pd.read_csv('djia_index.csv').iloc[:,1]


pca = PCA(10)
pca.fit(data)
pca.explained_variance_ratio_
maxWeightInd = pca.components_[0].argmax()
print('Company with the highest impact to the 1st component is', \
      list(data)[maxWeightInd])

totRat = 0
for i in range(len(pca.explained_variance_ratio_)):
    totRat += pca.explained_variance_ratio_[i]
    if totRat>=0.90:
        print('Sufficient number of components for 90% e.v. =',i+1)
        break

result = pca.transform(data)
comp1 = result[:,0]

print('Pearson corr between the 1st component and DJ index =', \
      numpy.corrcoef(comp1,dj)[0,1])
