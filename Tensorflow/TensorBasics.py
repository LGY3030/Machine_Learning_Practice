#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf


# In[2]:


print(tf.__version__)


# In[26]:


#Initialization of tensors
x = tf.constant(10,dtype=tf.float32)
print(x)
x = tf.constant([[1,2,3],[4,5,6]])
print(x)
x = tf.ones((3,6))
print(x)
x = tf.zeros((2,3))
print(x)
x = tf.eye(6) #identity matrix
print(x)
x = tf.random.normal((3,8,4), mean=0, stddev=1)
print(x)
x = tf.random.uniform((5,5), minval=0, maxval=1)
print(x)
x = tf.range(9)
print(x)
x = tf.range(start=0, limit=9, delta=2, dtype=tf.float32)
print(x)
x = tf.range(start=0, limit=9, delta=2)
x = tf.cast(x, dtype=tf.float32)
print(x)


# In[58]:


#Mathematical Operations
x = tf.constant([1,2,3])
y = tf.constant([9,8,7])
z = tf.add(x,y)
print(z)
w = x+y
print(w)
z = tf.subtract(x,y)
print(z)
w = x-y
print(w)
z = tf.divide(x,y)
print(z)
w = x/y
print(w)
z = tf.multiply(x,y)
print(z)
w = x*y
print(w)
z = tf.tensordot(x,y,axes=1) #axes是指x的最後n個維度 與 y的前n個維度相乘
print(z)
w = tf.reduce_sum(x*y, axis=0) #axis是指第幾個維度的總和
print(w)
z = x**5
print(z)

x = tf.random.normal((2,3))
y = tf.random.normal((3,4))
z = tf.matmul(x,y)
print(z)
w = x@y
print(z)


# In[13]:


#Indexing
x = tf.constant([1,2,3,4,5,1,2,3,4,5,3,2,1])
print(x[:])
print(x[1:])
print(x[1:3])
print(x[::2])#隔一個
print(x[::-1])#reverse

indices = tf.constant([1,3])
x_ind = tf.gather(x,indices)
print(x_ind)

x = tf.random.normal((2,3))
print(x[0])
print(x[0,:])
print(x[0:2])
print(x[0:2,:])


# In[24]:


#Reshaping
x = tf.range(9)
print(x)

x = tf.reshape(x, (3,3))
print(x)

x = tf.transpose(x, perm=[1,0])
print(x)


# In[ ]:




