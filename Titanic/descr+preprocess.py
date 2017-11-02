# data description and processing

import pandas as pd
import os
cwd   = os.chdir('D:\Programming\Python\IntroML\Titanic')
#os.getcwd()

data = pd.read_csv('titanic.csv', sep = '\t', index_col='PassengerId')
data.head()
data[:10]


fm = data['Sex'].value_counts()
print('number of men =',fm[0],'\nnumber of women =', fm[1])

sur = data['Survived'].value_counts()
percSur = sur[1]*100/(sur[0]+sur[1])
print('percent of survived =', round(percSur,2))

age = data['Age']
print('age mean =',round(age.mean(),2),'\nage median =',round(age.median(),1))

data1 = data[['SibSp', 'Parch']]
pcor = data1.corr(method='pearson')
print('pearson corr between SibSp and Parch =',round(pcor['SibSp'][1],2))

data2 = data.loc[data['Sex'] == 'female']
name = data2['Name'].copy()
rows = name.index.tolist()

for row in rows:
    if name.at[row][len(name.at[row])-1] ==' ': 
        name.at[row]=name.at[row][:len(name.at[row])-1]
    if 'Miss' in name.at[row]:
        name.at[row] = name.at[row][name.at[row].index('.')+2:len(name.at[row])]
    elif '(' in name.at[row]:
        if name.at[row].rfind(' ')>name.at[row].index('('):
            name.at[row] = name.at[row][name.at[row].index('(')+1:name.at[row].rfind(' ')]
        else:
            name.at[row] = name.at[row][name.at[row].index('(')+1:name.at[row].rfind(')')]
    else:
        name.at[row] = name.at[row][name.at[row].index('.')+2:len(name.at[row])]
    if '"' in name.at[row]:
        name.at[row] = name.at[row][:name.at[row].index('"')-1]

nf = name.value_counts()
print('the most frequent name is',nf.idxmax())
