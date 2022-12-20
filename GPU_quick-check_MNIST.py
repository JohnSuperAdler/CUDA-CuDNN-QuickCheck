#!/usr/bin/env python
# coding: utf-8

# ## 讀取模組、建立函數、建立參數

# In[1]:


##### Module Import

### Common Module

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

### Tensorflow Module

import tensorflow as tf
from keras.datasets import mnist
from keras.utils    import np_utils
from keras.models   import Sequential
from keras.layers   import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


# In[2]:


##### Functions

### for plotting train history
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    plt.show()


# In[3]:


##### Parameter

### VRAM Usage: Input VRAM memory usage limitation in MB
mem_limit = 1024 * 4


# ## 檢查tensorflow是否有抓到GPU，顯示0表示沒抓到。

# In[4]:


##### Check if GPU detected

gpus = tf.config.list_physical_devices('GPU')
if gpus:  
    try: # Restrict TensorFlow to only allocate "mem_limit" MB of memory on the first GPU
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=mem_limit)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f'{len(gpus)} Physical GPU(s) detected.')
        print(f'{len(logical_gpus)} Logical GPU(s) detected.')
    except RuntimeError as e: # Virtual devices must be set before GPUs have been initialized
        print(e)


# ## 讀取資料、前處理

# In[5]:


##### Load MNIST dataset from internet via mnist.load_data()

(x_Train, y_Train), (x_Test, y_Test) = mnist.load_data()


# In[6]:


##### Data Pre-processing

# Reshape
x_Train4D = x_Train.reshape(x_Train.shape[0], 28, 28, 1).astype('float32')
x_Test4D  = x_Test.reshape(x_Test.shape[0], 28, 28, 1).astype('float32')

# Normalization
x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize  = x_Test4D / 255

# Change to One-Hot format
y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot  = np_utils.to_categorical(y_Test)


# ## 建立模型、執行訓練

# In[7]:


##### Create Model

model = Sequential(name='MNIST_GPU_test')

model.add(Conv2D(filters=16,
                 kernel_size=(5, 5),
                 padding='same',
                 input_shape=(28, 28, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36,
                kernel_size=(5, 5),
                padding='same',
                activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# In[8]:


##### Print model summary

print(model.summary())


# ### 檢查是否有使用GPU計算：
# ### 執行下方model.fit時，GPU中的Cuda選項有在動才是真的在用GPU計算。
# 
# 
# ![GPU%20running-3.png](attachment:GPU%20running-3.png)

# In[9]:

##### Announce

print()
print('#####################################################################')
print('###                                                               ###')
print('###   NOTICE: Remenber to check GPU CUDA usage in Task Maneger.   ###')
print('###                                                               ###')
print('#####################################################################')
print()


##### Model compiling

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_history = model.fit(x=x_Train4D_normalize,
                          y=y_TrainOneHot,
                          validation_split=0.2,
                          epochs=10,
                          batch_size=30,
                          verbose=2)


# ## 訓練結果

# In[10]:


#show_train_history(train_history, 'accuracy', 'val_accuracy')


# In[11]:


#show_train_history(train_history, 'loss', 'val_loss')


# In[12]:


##### Test values

loss_test, matrices_test = model.evaluate(x_Test4D_normalize, y_TestOneHot)
print(f'Loss value of test: {loss_test}')
print(f'Matrices value of test: {matrices_test}')


# In[13]:


##### Get prediction of x_Test

prediction_mx = model.predict(x_Test4D_normalize)
classes       = np.argmax(prediction_mx, axis=1)

print(f'First 10 prediction: {classes[:10]}')


# In[14]:


##### Confusion Matrix

print(pd.crosstab(y_Test, classes, rownames=['label'], colnames=['predict']))


# ## 結束

print()
print(f'{len(gpus)} Physical GPU(s) detected.')
print(f'{len(logical_gpus)} Logical GPU(s) detected.')
print('Done')