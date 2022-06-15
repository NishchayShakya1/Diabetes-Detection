#!/usr/bin/env python
# coding: utf-8

# # DIABETES PREDICTION

# In[17]:


import numpy as np
import pandas as pd


# # IMPORTING DATASET

# In[18]:


dataset = pd.read_csv("diabetes.csv")
dataset.head(11)


# # Analyzing dataset

# In[19]:


dataset.info()


# In[20]:


dataset.isnull().sum()


# In[21]:


X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]


# # Splitting Data

# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 25, random_state = 0)


# # Applying classifiers AND evaluation

# # RANDOM FOREST

# In[23]:


from sklearn.ensemble import RandomForestClassifier
classifier =  RandomForestClassifier(n_estimators = 6, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# In[25]:


from sklearn.metrics import accuracy_score
acc_logreg2 = round(accuracy_score(y_pred, y_test), 2)*100
print("Accuracy : ",acc_logreg2)


# # LOGISTIC REGRESSION

# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, classification_report
logreg = LogisticRegression(solver = 'lbfgs', max_iter = 1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_logreg1 = round(accuracy_score(y_pred, y_test), 2)*100
print("Accuracy : ",acc_logreg1)


# # K NEIGHBOUR CLASSIFIER

# In[29]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_knn = round(accuracy_score(y_pred, y_test), 2) * 100
print("Accuracy : ", acc_knn)

