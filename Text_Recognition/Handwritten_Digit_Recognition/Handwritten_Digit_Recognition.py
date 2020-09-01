#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import fetch_openml


# In[3]:


mnist = fetch_openml('mnist_784')


# In[5]:


x, y = mnist['data'], mnist['target']


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


import matplotlib
import matplotlib.pyplot as plt


# In[8]:


some_digit = x[2500]
some_digit_image = some_digit.reshape(28,28)


# In[10]:


plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation = 'nearest')


# In[ ]:




