
# coding: utf-8
#Author:Farzad Fakhari-Tehrani

# In[28]:


from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt  

xm = np.array([0,1,2,3,4,5])
ym = np.array([0.1,0.2,0.3,0.5,1.0,0.9])


# In[29]:


m=GEKKO()


# In[30]:


m.x=m.Param(value=np.linspace(-1,6))


# In[37]:


m.y=m.Var()


# In[32]:


m.options.IMODE=2


# In[38]:


m.cspline(m.x,m.y,xm,ym)


# In[39]:


m.options.IMODE=2


# In[40]:


m.solve(disp=False)


# In[41]:


plt.plot(m.x.value,m.y.value, label='cubic spline')


# In[36]:


get_ipython().magic('matplotlib inline')


# In[15]:




