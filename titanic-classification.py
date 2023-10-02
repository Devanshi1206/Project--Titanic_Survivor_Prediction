#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Classification

# In[1]:


get_ipython().system('python --version')


# In[2]:


import numpy as np 
import pandas as pd 


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore') # To prevent kernel from showing any warning


# ## Importing Data

# In[4]:


df=pd.read_csv('titanic.csv')
df.head()


# Survived : 0 = Dead 1 = Alive
# 
# Pclass : 1 = First class 2 = Second class 3 = Third class
# 
# SibSp : Number of siblings
# 
# Parch : Parent and Children
# 
# C = Cherbourg Q = Queenstown S = Southampton

# In[5]:


print(df.shape)
print("")
df.info()


# In[6]:


# data visualisation before cleaning data 

plt.figure(figsize=(14,14))
plt.subplot(3,2,1)
sns.boxplot(x='Sex', y = 'Age',data= df)
# 'male': 1, 'female': 0

plt.subplot(3,2,2)
sns.distplot(df['Fare'],color='g')

plt.subplot(3,2,3)
sns.distplot(df['Age'],color='g')

plt.subplot(3,2,4)
sns.countplot(x='Sex', data=df)

plt.subplot(3,2,5)
sns.histplot(df['Age'])


plt.tight_layout()
plt.show()


# ## Data Cleaning

# In[7]:


df.drop(columns=['PassengerId','Name','Embarked','Cabin'],inplace=True)


# In[8]:


df.head()


# In[9]:


df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

df['Sex'].value_counts()


# In[10]:


df.isna().sum()


# In[11]:


df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())


# In[12]:


df.isna().sum()


# In[13]:


import re
df['Ticket'] = df['Ticket'].apply(lambda x: re.sub(r'\D', '', x))


# In[14]:


df.info()


# In[15]:


df.describe()


# ## Data Visualisation

# In[16]:


plt.figure(figsize=(14,14))
plt.subplot(3,2,1)
sns.boxplot(x='Sex', y = 'Age',data= df)
# 'male': 1, 'female': 0

plt.subplot(3,2,2)
sns.distplot(df['Fare'],color='g')

plt.subplot(3,2,3)
sns.distplot(df['Age'],color='g')

plt.subplot(3,2,4)
sns.countplot(x='Sex', data=df)

plt.subplot(3,2,5)
sns.histplot(df['Age'])


plt.tight_layout()
plt.show()


# ##### Note: 'male': 1, 'female': 0

# In[17]:


sns.countplot(x='Pclass',data=df,hue='Survived')
legend_labels = {0: 'Not Survived', 1: 'Survived'}
plt.legend(title='Survived', labels=[f"{key} = {value}" for key, value in legend_labels.items()])
plt.title('diffrent Pclass by Survival Status')
plt.show()


# In[18]:


sns.histplot(data=df[df['Survived'] == 1], x='Age', kde=True, color='blue', label='Survived')
sns.histplot(data=df[df['Survived'] == 0], x='Age', kde=True, color='red', label='Not Survived')

plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Age Distribution by Survival Status')

plt.legend(title='Survival Status')

plt.show()


# In[19]:


plt.figure(figsize=[12,8])
sns.heatmap(df.corr(),annot=True)


# ## ML Modeling

# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


x = df.drop(['Survived'],axis=1)
y = df['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 30)


# In[22]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## Model Building

# In[23]:


models = {'Logistic Regression': LogisticRegression(),
          'Decision Tree': DecisionTreeClassifier(),
          'Random Forest': RandomForestClassifier(),
          'Support Vector Machine': SVC(),
          'Gradient Boosting': GradientBoostingClassifier(),
          'K-Nearest Neighbors': KNeighborsClassifier(),
          'Neural Network': MLPClassifier(max_iter=1000)  
          }


# In[24]:


for model_name, model in models.items():
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred)  # Compute classification report
    print(f"{model_name} - Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{report}\n")


# ##### Only 3 models (Decision Tree, Random Forest, Gradient Boosting) out of 6 models proviedes 100% accuracy (i.e 1.0)

# ## ML Modeling

# In[25]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
rand = RandomForestClassifier()
normal = MinMaxScaler()


# In[26]:


normal_fit = normal.fit(X_train)
new_xtrain = normal_fit.transform(X_train)
new_xtest = normal_fit.transform(X_test)


# In[27]:


fit_rand = rand.fit(new_xtrain, Y_train)
#predicting score
rand_score = rand.score(new_xtest, Y_test)
print('Score of model is : ', rand_score*100,'%')


# In[28]:


X_predict = list(rand.predict(X_test))
predicted_df = {'predicted_values': X_predict,'original_values': Y_test}
print(pd.DataFrame(predicted_df).head(10))
print('')
print('Here O = not survived and 1 = survived')


# ##### Random Forest Classifier predicts all the survivers correctly.
