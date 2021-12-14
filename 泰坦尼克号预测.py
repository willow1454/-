#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


train_data = pd.read_csv("F:/课程/5.大三上课程/机器学习/课程设计/titanic/train.csv")
test_data = pd.read_csv("F:/课程/5.大三上课程/机器学习/课程设计/titanic/test.csv")


# In[3]:


train_data.head()


# In[4]:


test_data.head()


# In[5]:


train_data.shape


# In[6]:


test_data.shape


# In[7]:


train_data.info()


# In[8]:


test_data.info()


# In[9]:


train_data.describe()


# In[10]:


test_data.describe()


# In[11]:


train_data.describe(include=['O'])


# In[12]:


test_data.describe(include=['O'])


# In[13]:


#查看缺失值
for feature in train_data.columns:
    print(feature,train_data[feature].isnull().sum(),'of',train_data.shape[0],'values 缺失')


# In[14]:


for feature in test_data.columns:
    print(feature,test_data[feature].isnull().sum(),'of',test_data.shape[0],'values are missing')


# In[15]:


#查看生存率和死亡率的比例和人数
survived = train_data[train_data['Survived']==1]
died = train_data[train_data['Survived']==0]

print('Survived %i (%.1f%%)'%(len(survived),len(survived)/len(train_data)*100))
print('Died %i (%.1f%%)'%(len(died),len(died)/len(train_data)*100))
print('Total %i '%(len(train_data)))


# In[16]:


#查看各个船舱等级的人数
train_data.Pclass.value_counts()


# In[17]:


train_data.groupby('Survived')['Pclass'].value_counts()


# In[18]:


train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean()


# In[20]:


import seaborn as sns
sns.countplot(x='Pclass', hue='Survived', data=train_data)


# In[22]:


sns.heatmap(train_data.corr(),cmap="YlGnBu")


# In[23]:


#查看性别的存活率
train_data.Sex.value_counts()


# In[24]:


train_data.groupby('Survived').Sex.value_counts()


# In[25]:


train_data[['Sex','Survived']].groupby('Sex').mean()


# In[26]:


sns.countplot(x = 'Sex', hue = 'Survived', data = train_data)


# In[27]:


tab = pd.crosstab(train_data['Pclass'], train_data['Sex'])
tab


# In[28]:


import os
import matplotlib.pyplot as plt
tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Pclass')
plt.ylabel('Percentage')


# In[29]:


sns.factorplot('Sex', 'Survived', hue='Pclass', size=4, aspect=2, data=train_data)


# In[30]:


sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=train_data)


# In[32]:


train_data.Embarked.value_counts()


# In[33]:


train_data.groupby('Embarked').Survived.value_counts()


# In[35]:


sns.countplot(x = 'Embarked',hue = 'Survived', data = train_data)


# In[36]:


train_data[['Embarked','Survived']].groupby('Embarked').mean()


# In[37]:


train_data['Parch'].value_counts()


# In[38]:


train_data.groupby('Survived').Parch.value_counts()


# In[39]:


sns.barplot(x = 'Parch',y = 'Survived', ci = None,data = train_data)


# In[40]:


train_data[['Parch','Survived']].groupby('Parch').mean()


# In[41]:


train_data.SibSp.value_counts()


# In[42]:


train_data.groupby('Survived').SibSp.value_counts()


# In[43]:


train_data[['SibSp','Survived']].groupby('SibSp').mean()


# In[44]:


sns.barplot(x = 'SibSp', y = 'Survived', ci = None, data = train_data)


# In[45]:


fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

sns.violinplot(x="Embarked", y="Age", hue="Survived", data=train_data, split=True, ax=ax1)
sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train_data, split=True, ax=ax2)
sns.violinplot(x="Sex", y="Age", hue="Survived", data=train_data, split=True, ax=ax3)


# In[46]:


total_survived = train_data[train_data['Survived']==1]
total_not_survived = train_data[train_data['Survived']==0]
male_survived = train_data[(train_data['Survived']==1) & (train_data['Sex']=="male")]
female_survived = train_data[(train_data['Survived']==1) & (train_data['Sex']=="female")]
male_not_survived = train_data[(train_data['Survived']==0) & (train_data['Sex']=="male")]
female_not_survived = train_data[(train_data['Survived']==0) & (train_data['Sex']=="female")]

plt.figure(figsize=[15,5])
plt.subplot(111)
sns.distplot(total_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(total_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Age')
plt.figure(figsize=[15,5])


plt.subplot(121)
sns.distplot(female_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(female_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Female Age')

plt.subplot(122)
sns.distplot(male_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(male_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Male Age')


# In[47]:


plt.figure(figsize=(15,6))
sns.heatmap(train_data.drop('PassengerId',axis=1).corr(), vmax=0.6, square= True, annot=True)


# In[48]:


train_test_data = [train_data, test_data] # combining train and test dataset

for data in train_test_data:
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.')


# In[49]:


test_data['Title'].head() 


# In[50]:


pd.crosstab(train_data['Title'], train_data['Sex'])


# In[51]:


for data in train_test_data:
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',   'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    
train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[52]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for data in train_test_data:
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)


# In[53]:


train_data.head()


# In[54]:


#女性用1，男性用0代替
title_mapping = {'female':1,'male':0}
for data in train_test_data:
    data['Sex'] = data['Sex'].map(title_mapping)


# In[55]:


#用常见值代替空值
for data in train_test_data:
    print(data.Embarked.value_counts())
    print(data.Embarked.isnull().sum())


# In[56]:


for data in train_test_data:
    data.Embarked = data.Embarked.fillna('S')


# In[57]:


for data in train_test_data:
    data.Embarked = data.Embarked.map({'S':0,'C':1,'Q':2})


# In[58]:


train_data.head()


# In[59]:


for data in train_test_data:
    age_mean = data.Age.mean()
    age_std = data.Age.std()
    age_null_value = data.Age.isnull().sum()
    
    age_train_data = np.random.randint(age_mean-age_std,age_mean+age_std,size = age_null_value)
    data['Age'][np.isnan(data.Age)] = age_train_data
    data['Age'] = data['Age'].astype(int)


# In[60]:


#年龄分成五组
train_data['AgeBand'] = pd.cut(train_data['Age'], 5)  

print (train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())


# In[61]:


for data in train_test_data:
    data.loc[ data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[ data['Age'] > 64, 'Age'] = 4


# In[62]:


train_data.head()


# In[63]:


for data in train_test_data:
    data.Fare[np.isnan(data.Fare)] = data.Fare.mean()


# In[64]:


#年龄分成五组
train_data['FareBand'] = pd.qcut(train_data['Fare'], 4)  

print (train_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())


# In[65]:


for data in train_test_data:
    data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
    data.loc[ data['Fare'] > 31, 'Fare'] = 3
    data['Fare'] = data['Fare'].astype(int)


# In[66]:


train_data.head()


# In[67]:


for data in train_test_data:
    data['Family'] = data['SibSp'] + data['Parch'] + 1
train_data[['Family','Survived']].groupby(['Family'],as_index = False).mean()


# In[68]:


for data in train_test_data:
    data['IsAlone'] = np.where(data['Family']==1,1,0)


# In[69]:


#单身与非单身的存活率
train_data[['IsAlone','Survived']].groupby(['IsAlone'],as_index = False).mean()


# In[70]:


train_data= train_data.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','AgeBand','FareBand','Family'],axis = 1)
test_data = test_data.drop(['Name','SibSp','Parch','Ticket','Cabin','Family'],axis = 1)


# In[71]:


X_train = train_data.drop('Survived',axis = 1)
y_train = train_data['Survived']
X_test = test_data.drop('PassengerId',axis = 1)


# In[72]:


#模型创建
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
models = [LogisticRegression(),SVC(),LinearSVC(),DecisionTreeClassifier(),          RandomForestClassifier(n_estimators = 100),KNeighborsClassifier(10),         SGDClassifier(max_iter = 100, tol = None),Perceptron(max_iter = 100, tol = None),GaussianNB()]


# In[73]:


models_list = ['Logistic Regression','Support Vector Machines','Linear Support Vector Machines',               'Decision Tree','Random Forest','k-Nearest Neighbours', 'Stochastic Gradient Descent',               'Perceptron','Naive Bayes']
accuracy_list = []


# In[74]:


for i in models:
    i.fit(X_train,y_train)
    y_pred_log = i.predict(X_test)
    accuracy = round(100*i.score(X_train,y_train),2)
    print(str(i) + ' Accuracy {}%'.format(accuracy))
    accuracy_list.append(accuracy)


# In[75]:


models_dataframe = pd.DataFrame({'Models': models_list,'Accuracy':accuracy_list})
models_dataframe.set_index('Models', inplace = True)
models_dataframe.sort_values(by = ['Accuracy'], ascending=False)


# In[76]:


model = RandomForestClassifier(n_estimators = 100)
model.fit(X_train,y_train)
y_pred_log = model.predict(X_train)
accuracy = round(100*model.score(X_train,y_train),2)
print(str(model) + ' Accuracy {}%'.format(accuracy))


# In[77]:


from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
conf_mat = confusion_matrix(y_train,y_pred_log)
rows = ['生存','死亡']
cols = ['预测生存','预测死亡']
conf_mat_frame = pd.DataFrame(conf_mat,index = rows, columns = cols)

np.set_printoptions(precision = 2)

print('数字中的混淆矩阵')
print(conf_mat)
print('')
print('')

print('以百分比表示的混淆矩阵')
conf_mat_perc = conf_mat.astype(float)/conf_mat.sum(axis = 1)[:,np.newaxis]
print(conf_mat_perc)
print('')
print('')

conf_mat_perc_frame = pd.DataFrame(conf_mat_perc,index = rows, columns = cols)

plt.figure.figsize = (15,5)

plt.subplot(121)
sns.heatmap(conf_mat_frame, annot = True, fmt='d')

plt.subplot(122)
sns.heatmap(conf_mat_perc_frame, annot = True)


# In[78]:


test_data.head()


# In[79]:


y_pred = model.predict(X_test)


# In[80]:


answer = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':y_pred})


# In[81]:


answer.to_csv('submission.csv',index = False)


# In[ ]:




