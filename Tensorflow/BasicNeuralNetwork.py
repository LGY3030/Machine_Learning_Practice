#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[5]:


x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0
print(x_train.shape)
print(x_test.shape)


# In[6]:


#x_train = tf.convert_to_tensor(x_train)


# In[30]:


#Sequential API (Very convenient, not very flexible) -> one to one
model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10),
    ]
)

'''
這種寫法可以在各行之間加print(model.summary()) 來debug
model = keras.Sequential()
model.add(keras.Input(shape=(28*28)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10))
'''

print(model.summary())

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),#上面最後一層沒有activation='softmax'
    optimizer = keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)


# In[7]:


#Functional API (more flexible)
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),#上面最後一層有activation='softmax'
    optimizer = keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)


# In[14]:


model = keras.Sequential()
model.add(keras.Input(shape=(28*28)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10))

model = keras.Model(inputs=model.inputs, outputs=[model.layers[-2].output])

feature = model.predict(x_train)
print(feature.shape)


# In[ ]:




