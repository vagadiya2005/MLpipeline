#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/MYoussef885/House_Price_Prediction/blob/main/House_Price_Prediction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import joblib
import pickle
import os


# In[2]:


import sys
get_ipython().system('{sys.executable} -m pip install sklearn')


import sys
get_ipython().system('{sys.executable} -m pip install xgboost')






# Importing the Boston House Price Dataset

# In[3]:


house_price_dataset = sklearn.datasets.fetch_california_housing()


# In[4]:


print(house_price_dataset)


# In[5]:


# Loading the dataset to a pandas dataframe
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns = house_price_dataset.feature_names)


# In[6]:


house_price_dataframe.head()


# In[7]:


# add the target column to the dataframe
house_price_dataframe['price'] = house_price_dataset.target


# In[8]:


house_price_dataframe.head()


# In[9]:


# checking the number of rows and columns in the dataframe
house_price_dataframe.shape


# In[10]:


# check for missing values
house_price_dataframe.isnull().sum


# In[11]:


# statistical measures of the dataset
house_price_dataframe.describe()


# Understanding the **correlation** between various features in the dataset

# 1. Positive Correlation
# 2. Negative Correlation

# In[12]:


correlation = house_price_dataframe.corr()


# In[13]:


# constructing a heatmap to understand the correlation

plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')


# Splitting the data and target

# In[14]:


X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']


# In[15]:


print(X,Y)


# Splitting the data into training data and test data

# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# In[17]:


print(X.shape, X_train.shape, X_test.shape)


# Model Training

# XGBoost Regressor

# In[18]:


# load the model
model = XGBRegressor()


# In[19]:


#training the model with X_train
model.fit(X_train, Y_train)


# Evaluation

# Prediction on training data

# In[20]:


# accuracy for prediction on training data
training_data_prediction = model.predict(X_train)


# In[21]:


print(training_data_prediction)


# In[22]:


# R Squared Error
score_1 = metrics.r2_score(Y_train, training_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print('R Sqaured Error:', score_1)
print('Mean Absolute Error:', score_2)


# Visualize the actuale prices and predicted prices

# In[23]:


plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()


# Prediction on test data

# In[24]:


# accuracy for prediction on test data
test_data_prediction = model.predict(X_test)


# In[25]:


# R Squared Error
score_1 = metrics.r2_score(Y_test, test_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print('R Sqaured Error:', score_1)
print('Mean Absolute Error:', score_2)


# In[26]:


# Save the trained model
with open('model.pkl', 'wb') as f:
   pickle.dump(model, f)
   print("Model trained and saved as model.pkl")

