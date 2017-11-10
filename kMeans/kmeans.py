# tested on Python 3.6.3
# work dir must contain: parrots.jpg
# compresses an image with kmeans and calculates PSNR metrics

import numpy  as np
import sklearn.cluster  # http://scikit-learn.org/stable/
import skimage          # http://scikit-image.org/
import skimage.io
#import os               # https://docs.python.org/3/library/os.html
import pylab            # https://scipy.github.io/old-wiki/pages/PyLab
import math             # https://docs.python.org/2/library/math.html

# set cd
#os.chdir('D:\Programming\Python\IntroML\kMeans')

# load image
image   =  skimage.io.imread('parrots.jpg')
#pylab.imshow(image)
# convert to float
imfloat = skimage.img_as_float(image)
dim     = np.shape(imfloat)[0:2]

# restructure data for kmeans
objfeat = np.zeros((dim[0]*dim[1],np.shape(imfloat)[2]))
k = 0
for i in range(dim[0]):
    for j in range(dim[1]):
        objfeat[k] = imfloat[i,j]
        k+=1

# fit kmeans and define clusters for our data
kmeans = sklearn.cluster.KMeans(n_clusters = 11, init='k-means++', \
                                random_state=241)
result = kmeans.fit_predict(objfeat)

# change the color of each cluster to its mean/median
objfeat_mean = np.zeros((dim[0]*dim[1],3))
objfeat_medi = np.zeros((dim[0]*dim[1],3))

for i in range(kmeans.n_clusters):
    cluster_i     = [x for x in range(len(result)) if result[x]==i]
    colors_i      = [objfeat[x] for x in cluster_i]
    color_i_mean  = [np.array(colors_i)[:,j].mean() for j in range(3)]
    color_i_medi  = [np.median(np.array(colors_i)[:,j]) for j in range(3)]
    for obj in cluster_i:
        objfeat_mean[obj] = color_i_mean
        objfeat_medi[obj] = color_i_medi

image_new_mean = np.zeros((dim[0],dim[1],3))
image_new_medi = np.zeros((dim[0],dim[1],3))

k = 0
for i in range(dim[0]):
    for j in range(dim[1]):
        image_new_mean[i,j] = objfeat_mean[k]
        image_new_medi[i,j] = objfeat_medi[k]
        k+=1

# show resulting images
pylab.imshow(image_new_mean)
pylab.show()
pylab.imshow(image_new_medi)
pylab.show()

# calculate mse for mean and median
dif_mean  = imfloat - image_new_mean
dif_medi  = imfloat - image_new_medi

mse_mean  = np.sum(dif_mean**2)/(dim[0]*dim[1]*3)
mse_medi  = np.sum(dif_medi**2)/(dim[0]*dim[1]*3)

PSNR_mean = 20*math.log10(1) - 10*math.log10(mse_mean)
PSNR_medi = 20*math.log10(1) - 10*math.log10(mse_medi)

print('PSNR for mean and median are', round(PSNR_mean,2),'and', round(PSNR_medi,2))
