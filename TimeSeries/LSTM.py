# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 20:17:15 2020
LSTM stands for long short term memory.


It is a model or architecture that extends the memory of recurrent neural
networks. Typically, recurrent neural networks have ‘short term memory’ in
that they use persistent previous information to be used in the current neural
network.

From: https://medium.com/ai-in-plain-english/time-series-forecasting-predicting-stock-prices-using-an-lstm-model-30d6f1ca2640
Tesla stock price data from: https://finance.yahoo.com/quote/GOOG/history?period1=1433548800&period2=1591833600&interval=1d&filter=history&frequency=1d
keras & tensor flow would not install in conda base, so installed in py37 instead.
Python 3’s f-Strings: An Improved String Formatting Syntax. https://realpython.com/python-f-strings/
Also: https://realpython.com/python-timer/  (didn't use here, but it's nice.)
https://stackoverflow.com/questions/45662253/can-i-run-keras-model-on-gpu
https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
http://cis.bentley.edu/sandbox/wp-content/uploads/Documentation-on-f-strings.pdf
KEY KNOWLEDGE:::
    https://stackoverflow.com/questions/59711670/tensorflow-backend-error-attributeerror-module-tensorflow-has-no-attribute
    
I recommend that you switch to using the keras module inside TensorFlow.

It is better maintained and you will not have incompatibility issues.

According to Francois Chollet:

Keras version 2.3.0 is here, and it is the last major multi-backend release. 
Going forward, users are recommended to switch their code over to:
    tf.keras in TensorFlow 2.0. 
    This release brings API changes and a few breaking changes. 
    Have a look under the hood and see what it includes, 
    as well as what the plans are going forward.

KEY COMMANDS:
    >nvcc -V
Other links: includes cuDNN cuda for Deep Neural Networks
    https://www.codingforentrepreneurs.com/blog/install-tensorflow-gpu-windows-cuda-cudnn/

11 Dec 2020: now running with Python 3.8.6 Tensorflow 2.3.0 BUT NO GPU!

My 1st Neural Network! 
RUNNING ON GPU: TAKES 42.8 SECONDS   
"""

# import math
import matplotlib
import matplotlib.pyplot as plt
# import keras  # deep learning lib for theano and tensorflow
import pandas as pd
import numpy as np
import os
import sys
import time

#import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
# from keras.layers import *
import sklearn
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split
# from keras.callbacks import EarlyStopping

from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())
print('Conda Envronment:  ', os.environ['CONDA_DEFAULT_ENV'])
print(f'Gpu  Support:       {tf.test.is_built_with_gpu_support()}')
print(f'Cuda Support:       {tf.test.is_built_with_cuda()}')
print(f'Tensor Flow:        {tf.version.VERSION}')
pver = str(format(sys.version_info.major) +'.'+ format(sys.version_info.minor)+'.'+ format(sys.version_info.micro))
print('Python version:      {}.'.format(pver)) 
print('The numpy version:   {}.'.format(np.__version__))
print('The panda version:   {}.'.format(pd.__version__))

print('SciKitLearn:    {}'.format(sklearn.__version__))
print('matplotlib:     {}'.format(matplotlib.__version__))
print('tensorflow:     {}'.format(tf.version.VERSION))

# from tensorflow.python.client import device_lib 
# gpus = device_lib.list_local_devices();
# gpuName = gpus[1].physical_device_desc[17:].replace(', pci bus id: 0000:01:00.0,','');

import GPUtil
gpus = GPUtil.getGPUs()

tf.config.experimental.list_physical_devices('GPU')

time.strftime('%c')
starttime = time.perf_counter()
df = pd.read_csv("D:\\dDocuments\\ML\\Python\\data\\TSLA.csv")
print('Number of rows and columns:', df.shape)
df.head(5)

training_set = df.iloc[:800, 1:2].values
test_set = df.iloc[800:, 1:2].values

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
# X_test = scaler.fit_transform(X_test)

# Creating a data structure with 60 time-steps and 1 output
X_train = []
y_train = []
# X_train.info(memory_usage='deep')
for i in range(60, 800):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# (740, 60, 1)

model = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True, 
               input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units=50))
model.add(Dropout(0.2))
# Adding the output layer
# The Keras "Dense" layer
model.add(Dense(units=1))
#model.add(Dense(units=1, activation='sigmoid'))
# Compiling the RNN
model.compile(optimizer='adam', loss='mean_squared_error')
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# Fitting the RNN to the Training set
epochs = 100
# History object...
mdlHistory = model.fit(X_train, y_train, epochs=epochs, batch_size=32);
# r        = tfmodel.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

# Evaluate the model - evaluate() returns loss and accuracy
# until i get accuracy, i can not use the following:
# print("Train score:", model.evaluate(X_train, y_train));


plt.plot(mdlHistory.history['loss'], label='loss')  # from Udemy course
#plt.plot(mdlHistory.history['val_loss'], label='val_loss')
plt.legend()
# plt.show()
# plt.plot(mdlHistory.history['accuracy'], label='Accuracy')
# plt.legend()
plt.show()

# Getting the predicted stock price of 2017
dataset_train = df.iloc[:800, 1:2]
dataset_test = df.iloc[800:, 1:2]
dataset_total = pd.concat((dataset_train, dataset_test), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 519):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test.shape)
# (459, 60, 1)


predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

endtime = time.perf_counter()
duration = endtime - starttime

modelstart = time.strftime('%c')

s2 = '''model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
epochs = 100
mdlHistory = model.fit(X_train, y_train, epochs=epochs, batch_size=32);'''


plt.style.use('classic')
# Visualising the results
plt.plot(df.loc[800:, 'Date'], dataset_test.values, color='red',
         label='Real TESLA Stock Price')
plt.plot(df.loc[800:, 'Date'], predicted_stock_price, color='blue',
         label='Predicted TESLA Stock Price')
plt.xticks(np.arange(0, 459, 50))
plt.title(f'My 1st Neural Network!    {modelstart}')
plt.xlabel('File: LSTM.py   Src: www.medium.com/Serafeim Loukas. TESLA Stock Prediction')
plt.xticks(rotation=15)
plt.ylabel('TESLA Stock Price')
plt.legend()

plt.text(20, 120,# transform=trans1,
         s=s2, wrap=True, ha='left', va='bottom',
         fontsize=9, bbox=dict(facecolor='yellow', alpha=0.5))
plt.text(230, 10,# transform=trans1,
         s=f'Python vers: {pver}\nmulti-layer LSTM recurrent neural network' + 
         '\nUsing Keras & Tensorflow\nafter anaconda re-install' +
         f'\n{epochs} epochs, Duration: {duration:3.2f} seconds' +
         f'\ntensorflow Ver: {tf.version.VERSION}' + 
         f'\n{gpus[0].name  }' +
         f' GPU Memory: {gpus[0].memoryTotal} Mb' +
         '\nCuda 11.1.relgpu',
         wrap=True, ha='left', va='bottom',
         fontsize=10, bbox=dict(facecolor='aqua', alpha=0.5))
plt.show()


# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# print('{:,}'.format(device_lib.list_local_devices()[1].memory_limit));
# device_lib.list_local_devices()[1].physical_device_desc[17:]


