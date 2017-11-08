# tested on Python 3.6.3
# work dir must contain: titanic.csv
# data description and preprocessing

import pandas    # http://pandas.pydata.org/
#import os        # https://docs.python.org/3/library/os.html

# set cd
#os.chdir('D:\Programming\Python\IntroML\DecisionTree')
#os.getcwd()

# load data from csv
data = pandas.read_csv('titanic.csv', sep = '\t', index_col='PassengerId')
# show the first rows of the data
data.head()

# count number of men/women
fm = data['Sex'].value_counts()
print('number of men =',fm[0],'\nnumber of women =', fm[1])

# calculate % of survived
sur     = data['Survived'].value_counts()
percSur = sur[1]*100/(sur[0]+sur[1])
print('percent of survived =', round(percSur,2))

# calculate mean and median of age
age = data['Age']
print('age mean =',round(age.mean(),2),'\nage median =',round(age.median(),1))

# find pearson correlation between number of siblings/spouses and number of 
# parents/children
data1 = data[['SibSp', 'Parch']]
pcor  = data1.corr(method='pearson')
print('Pearson corr between SibSp and Parch =',round(pcor['SibSp'][1],2))

# define the most frequent female name
data2 = data.loc[data['Sex'] == 'female']
name  = data2['Name'].copy()
rows  = name.index.tolist()

# string transformation to isolate the actual name
for row in rows:
    if name.at[row][len(name.at[row])-1] ==' ': 
        name.at[row] = name.at[row][:len(name.at[row])-1]
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
print('The most frequent name is',nf.idxmax())
