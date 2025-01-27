#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10


# In[3]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[4]:


x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255


# In[11]:


model = keras.Sequential(
    [
        keras.Input(shape=(32,32,3)),
        #padding "valid" will reduce size, "same" will remain size
        #padding default is "valid"
        #pool_size default is (2,2)
        layers.Conv2D(32, (3,3), padding='valid', activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10),
    ]
)
print(model.summary())
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=3e-4),
    metrics=["accuracy"],
)
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
#model.evaluate(x_test, y_test, batch_size=64, verbose=2)


# In[12]:


model.evaluate(x_test, y_test, batch_size=64, verbose=2)


# In[15]:


def my_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x= layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# In[16]:


model = my_model()
print(model.summary())
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=3e-4),
    metrics=["accuracy"],
)
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)


# In[ ]:




