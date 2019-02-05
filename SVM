
# coding: utf-8
#Author: Farzad Fakhari-Tehrani
#SVM

# In[2]:


import numpy as np


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


from sklearn import svm


# In[10]:


from sklearn.datasets.samples_generator import make_blobs


# In[11]:


x,y= make_blobs(n_samples=40, centers=2, random_state=20)


# In[12]:


clf= svm.SVC(kernel='linear', C=1)


# In[13]:


clf.fit(x,y)


# In[14]:


plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=plt.cm.Paired)


# In[15]:


plt.show()


# In[23]:


plt.gca()


# In[22]:


get_ipython().magic('matplotlib inline')


# In[24]:


pickAFile()

