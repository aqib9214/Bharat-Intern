#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#read the dataset
iris=pd.read_csv("C:\\Users\\Asus\\Downloads\\IRIS.csv")


# In[3]:


iris.head(10)


# In[4]:


iris.shape


# In[5]:


list(iris.columns)


# In[6]:


#checking for null values
iris.isnull().sum()


# In[7]:


iris['species'].value_counts()


# In[8]:


n = len(iris[iris['species'] == 'Iris-versicolor'])
print("No of Versicolor in Dataset:",n)


# In[9]:


n1 = len(iris[iris['species']=='Iris-setosa'])
print("No of setosa in dataset:",n)


# In[10]:


n2 = len(iris[iris['species']=='Iris-virginica'])
print("No of virginica in dataset:",n)


# In[11]:


#Checking for outliars
import matplotlib.pyplot as plt
plt.figure(1)
plt.boxplot([iris['sepal_length']])
plt.figure(2)
plt.boxplot([iris['sepal_width']])
plt.show()


# In[12]:


iris.hist()
plt.figure(figsize=(10,7))
plt.show()


# In[13]:


sns.pairplot(iris,hue='species')


# In[14]:


#spliting the dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics


# In[15]:


train, test = train_test_split(iris, test_size = 0.3)
print(train.shape)
print(test.shape)


# In[16]:


train_X = train[['sepal_length', 'sepal_width', 'petal_length',
                 'petal_width']]
train_y = train.species

test_X = test[['sepal_length', 'sepal_width', 'petal_length',
                 'petal_width']]
test_y = test.species


# In[17]:


train_X


# In[18]:


train_y


# Using ML Model

# Using Logistic Regression

# In[19]:


model1 = LogisticRegression()
model1.fit(train_X, train_y)
prediction = model1.predict(test_X)
prediction


# In[20]:


print('Accuracy:',metrics.accuracy_score(prediction,test_y))


# In[21]:


#Confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
confusion_mat = confusion_matrix(test_y,prediction)
print("Confusion matrix: \n",confusion_mat)


# Using SVM

# In[22]:


from sklearn.svm import SVC
model1 = SVC()
model1.fit(train_X,train_y)

pred_y = model1.predict(test_X)

from sklearn.metrics import accuracy_score


# In[23]:


print("Acc=",accuracy_score(test_y,pred_y))


# Using KNN

# In[24]:


from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(train_X,train_y)
y_pred2 = model2.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred2))


# Using Naive Bayes

# In[25]:


from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(train_X,train_y)
y_pred3 = model3.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred3))


# Result of all the models

# In[26]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines', 'KNN','Naive Bayes'],
    'Score': [0.9777,0.9777,0.9555,0.9555]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)


# In[ ]:




