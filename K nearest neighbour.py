#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("Social_Network_Ads (1).csv")


# In[4]:


df.shape


# In[5]:


x = df.iloc[:,[2,3]].values


# In[6]:


y = df.iloc[:,4].values


# In[7]:


x


# In[8]:


y


# ## Train Test Split

# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)


# In[12]:


x_train


# In[13]:


x_test


# In[14]:


y_train


# In[15]:


y_test


# ## Apply Feature Scaling

# In[17]:


from sklearn.preprocessing import StandardScaler


# In[18]:


sc = StandardScaler()


# In[19]:


x_train = sc.fit_transform(x_train)


# In[20]:


x_test = sc.fit_transform(x_test)


# In[21]:


x_train


# In[22]:


x_test


# ## KNN Model Making

# In[23]:


from sklearn.neighbors import KNeighborsClassifier


# In[26]:


knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)


# In[27]:


y_pred


# In[28]:


y_test


# In[31]:


x_test[0]


# In[32]:


x_test[:,0]


# ## Visualizing predicted and tested data

# In[33]:


plt.scatter(x_test[:,0],y_test, c = y_pred)


# ## Calculating accuracy and confusion metrix

# In[35]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[36]:


print("Accuracy_Score : ",accuracy_score(y_test, y_pred))


# In[40]:


cf = confusion_matrix(y_test, y_pred)


# In[41]:


cf


# In[ ]:




