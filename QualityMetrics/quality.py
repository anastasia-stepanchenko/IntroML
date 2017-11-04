import pandas as pd
import sklearn
import os
os.chdir('D:\Programming\Python\IntroML\QualityMetrics')
data = pd.read_csv('classification.csv')

TP = TN = FP = FN = 0
for  i in range(len(data)):
    if data['true'][i] == 1: 
        if data['pred'][i] == 1: TP+=1
        else: FN+=1
    else:
        if data['pred'][i] == 0: TN+=1
        else: FP+=1
        
print('TP =',TP,'\nTN =',TN,'\nFP =',FP,'\nFN =',FN)

print('Accuracy =', sklearn.metrics.accuracy_score(data['true'],data['pred']))
print('Precision =',sklearn.metrics.precision_score(data['true'],data['pred']))
print('Recall  =',  sklearn.metrics.recall_score(data['true'],data['pred']))
print('F1 =',       sklearn.metrics.f1_score(data['true'],data['pred']))


#___________________

data2 = pd.read_csv('scores.csv')
roc = [sklearn.metrics.roc_auc_score(data['true'], \
                                     data2.iloc[:,x+1]) for x in range(4)]
print('AUC-ROC for all the methods:', roc)

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
