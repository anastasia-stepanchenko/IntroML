# tested on Python 3.6.3
# work dir must contain: classification.csv, scores.csv
# computes different quality metrics

import pandas           # http://pandas.pydata.org/
import sklearn.metrics  # http://scikit-learn.org/stable/
#import os               # https://docs.python.org/3/library/os.html

# set cd
#os.chdir('D:\Programming\Python\IntroML\QualityMetrics')
# load data from csv
data = pandas.read_csv('classification.csv')

# calculate TP, TN, FP, FN
TP = TN = FP = FN = 0
for  i in range(len(data)):
    if data['true'][i] == 1: 
        if data['pred'][i] == 1: TP+=1
        else: FN+=1
    else:
        if data['pred'][i] == 0: TN+=1
        else: FP+=1
        
print('TP =',TP,'\nTN =',TN,'\nFP =',FP,'\nFN =',FN)

# calculate accur, precision, recall anf F1 metrics
print('Accuracy  =', round(sklearn.metrics.accuracy_score(data['true'],data['pred']),2))
print('Precision =',round(sklearn.metrics.precision_score(data['true'],data['pred']),2))
print('Recall    =',  round(sklearn.metrics.recall_score(data['true'],data['pred']),2))
print('F1        =',       round(sklearn.metrics.f1_score(data['true'],data['pred']),2))


#___________________

# load another data from csv
data2 = pandas.read_csv('scores.csv')

# calculate AUC-ROC
roc = [sklearn.metrics.roc_auc_score(data['true'], \
                                     data2.iloc[:,x+1]) for x in range(4)]
print('AUC-ROC for all the methods:', roc)

# define scores with maximum precision if recall >= 0.7
pr = [sklearn.metrics.precision_recall_curve(data['true'], \
                                    data2.iloc[:,x+1]) for x in range(4)]
maxim = 0
iMax = -1
for i in range(len(pr)):
    for j in range(len(pr[i])):
        if pr[i][1][j]>=0.7:
            if maxim < pr[i][0][j]:
                maxim = pr[i][0][j]
                iMax = i

print('Scores with maximum precision if recall >= 0.7 is', list(data2)[iMax+1])
