#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ignore tensorflow messages
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255


# In[3]:


model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(layers.SimpleRNN(512, return_sequences=True, activation='relu'))
model.add(layers.SimpleRNN(512, activation='relu'))
model.add(layers.Dense(10))
model.summary()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)


# In[5]:


model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(layers.GRU(256, return_sequences=True, activation='relu'))
model.add(layers.GRU(256, activation='relu'))
model.add(layers.Dense(10))
model.summary()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)


# In[6]:


model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(layers.LSTM(512, return_sequences=True, activation='relu'))
model.add(layers.LSTM(512, activation='relu'))
model.add(layers.Dense(10))
model.summary()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)


# In[7]:


model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(
    layers.Bidirectional(
        layers.LSTM(512, return_sequences=True, activation='relu')
    )
)
model.add(
    layers.Bidirectional(
        layers.LSTM(512, activation='relu')
    )
)
model.add(layers.Dense(10))
model.summary()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)


# In[ ]:




