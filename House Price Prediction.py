#!/usr/bin/env python
# coding: utf-8

# HOUSE PRICE PREDICTION

# In[1]:


#import the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#import dataset
HouseDF = pd.read_csv("C:/Users/Asus/Downloads/Housing.csv")


# In[3]:


HouseDF.head()


# In[4]:


HouseDF.info()


# In[5]:


HouseDF.describe()


# In[6]:


HouseDF.columns


# EDA

# In[7]:


sns.pairplot(HouseDF)


# In[8]:


sns.distplot(HouseDF['price'])


# In[9]:


sns.heatmap(HouseDF.corr(), annot=True)


# Training a Linear Regression Model
# 
# X and y List

# In[10]:


X=HouseDF[['area','bathrooms','stories','parking']]
y=HouseDF['price']


# Split Data into Train, Test

# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# Creating and Training the LinearRegression Model

# In[13]:


from sklearn.linear_model import LinearRegression


# In[14]:


lm = LinearRegression()


# In[15]:


lm.fit(X_train,y_train)


# LinearRegression Model Evaluation

# In[16]:


print(lm.intercept_)


# In[17]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# Predictions from our Linear Regression Model

# In[18]:


predictions = lm.predict(X_test)


# In[19]:


plt.scatter(y_test,predictions)


# In[20]:


sns.distplot((y_test-predictions),bins=50);


# Regression Evaluation Metrics

# In[21]:


from sklearn import metrics


# In[22]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




