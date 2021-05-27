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
"""

# import math
import matplotlib
import matplotlib.pyplot as plt
# import keras  # deep learning lib for theano and tensorflow
import pandas as pd
import numpy as np
# import os
import sys
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
# from keras.layers import *
import sklearn
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split
# from keras.callbacks import EarlyStopping


pver = str(format(sys.version_info.major) + '.' +
           format(sys.version_info.minor) + '.' +
           format(sys.version_info.micro))
print('Python version: {}'.format(pver))
print('SciKitLearn:    {}'.format(sklearn.__version__))
print('matplotlib:     {}'.format(matplotlib.__version__))

starttime = time.perf_counter()
df = pd.read_csv("..\\data\\TSLA.csv")
print('Number of rows and columns:', df.shape)
df.head(5)

training_set = df.iloc[:800, 1:2].values
test_set = df.iloc[800:, 1:2].values

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
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
model.add(LSTM(units=50,
               return_sequences=True, input_shape=(X_train.shape[1], 1)))
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
model.add(Dense(units=1))
# Compiling the RNN
model.compile(optimizer='adam', loss='mean_squared_error')
# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs=100, batch_size=32)


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

plt.style.use('classic')
# Visualising the results
plt.plot(df.loc[800:, 'Date'], dataset_test.values, color='red',
         label='Real TESLA Stock Price')
plt.plot(df.loc[800:, 'Date'], predicted_stock_price, color='blue',
         label='Predicted TESLA Stock Price')
plt.xticks(np.arange(0, 459, 50))
plt.title('TESLA Stock Price Prediction')
plt.xlabel('Time')
plt.xticks(rotation=15)
plt.ylabel('TESLA Stock Price')
plt.legend()
plt.text(20, 120,# transform=trans1,
         s='Python vers: 3.7.8\nmulti-layer LSTM recurrent neural network' + 
         '\nUsing Keras & Tensorflow\n27 Nov 2020 after anaconda re-install' +
         f'\n100 epochs, Duration: {duration:3.2f} seconds',
         wrap=True, ha='left', va='bottom',
         fontsize=12, bbox=dict(facecolor='aqua', alpha=0.5))

plt.show()