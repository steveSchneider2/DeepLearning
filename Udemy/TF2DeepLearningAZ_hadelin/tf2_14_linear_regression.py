# -*- coding: utf-8 -*-
"""TF2.0 Linear Regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tNJWZ362FkX6skgYf3_Lo07k_-F6uqwr

"""
import time

starttime = time.perf_counter()
modelstart = time.strftime('%c')


# Commented out IPython magic to ensure Python compatibility.
# Install TensorFlow
# !pip install -q tensorflow-gpu==2.0.0-beta1

# try:
# #   %tensorflow_version 2.x  # Colab only.
#     except Exception:
#         pass

import tensorflow as tf
print(tf.__version__)
tf.config.experimental.list_physical_devices('GPU')

# Other imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import wget  # conda installed pywget...
# Get the data
url = 'https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv'
wget.download(url)
# !wget https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv

# Load in the data
data = pd.read_csv('moore.csv', header=None).values
type(data)  #ndarray ..
data.shape  # 162,2
data[:,0].shape  # ndarray ... a 1 dimensional array of length N
type(data[:,0])
X = data[:,0].reshape(-1, 1) # make it a 2-D array of size N x D where D = 1
X.shape     # 162, 1
Y = data[:,1]

# Plot the data - it is exponential!
plt.scatter(X, Y)
plt.show()

# Since we want a linear model, let's take the log
Y = np.log(Y)
plt.scatter(X, Y)
plt.show()
# that's better

# Let's also center the X data so the values are not too large
# We could scale it too but then we'd have to reverse the transformation later
#%% Normalize the data
X = X - X.mean()

#%% Tensorflow model
model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(1,)),
  tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,
                                                momentum=0.9),
              loss='mse')
# model.compile(optimizer='adam', loss='mse')


# learning rate scheduler
def schedule(epoch, lr):
  if epoch >= 50:
    return 0.0001
  return 0.001
 

scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

#%% Train the model
r = model.fit(X, Y, epochs=200, callbacks=[scheduler])

import pydot
import graphviz
from tensorflow import keras
# from tensorflow.keras import layers
keras.utils.plot_model(model, show_shapes=True)


endtime = time.perf_counter()
duration = round(endtime - starttime,3)
# Plot the loss
plt.plot(r.history['loss'], label='loss')
plt.legend()
plt.title(f'Moores Law {modelstart}\nTF2.0 Linear Regression Lecture 14')
# plt.xlabel('Udemy TensorFlow 2.0 Duration: ')
# plt.xlabel(f'Udemy TensorFlow 2.0 Duration: {duration}')
plt.show()
# Get the slope of the line
# The slope of the line is related to the doubling rate of transistor count
print(model.layers) # Note: there is only 1 layer, the "Input" layer doesn't count
print(model.layers[0].get_weights())

# The slope of the line is:
a = model.layers[0].get_weights()[0][0,0]

"""Our original model for exponential growth is:
$$ C = A_0 r^t $$
Where $ C $ is transistor the count and $ t $ is the year.
$ r $ is the rate of growth. For example, when $ t $ goes from 1 to 2, $ C $ increases by a factor of $ r $. When $ t $ goes from 2 to 3, $ C $ increases by a factor of $ r $ again.
When we take the log of both sides, we get:
$$ \log C = \log r * t + \log A_0 $$
This is our linear equation:
$$ \hat{y} = ax + b $$
Where:
$$ \hat{y} = \log C $$
$$ a = \log r $$
$$ x = t $$
$$ b = \log A_0 $$
We are interested in $ r $, because that's the rate of growth. Given our regression weights, we know that:
$$ a = 0.34188038 $$
so that:
$$ r = e^{0.34188038} = 1.4076 $$
To find the time it takes for transistor count to double, we simply need to find the amount of time it takes for $ C $ to increase to $ 2C $.
Let's call the original starting time $ t $, to correspond with the initial transistor count $ C $.
Let's call the end time $ t' $, to correspond with the final transistor count $ 2C $.
Then we also have:
$$ 2C = A_0 r ^ {t'} $$
Combine this with our original equation:
$$ C = A_0 r^t $$
We get (by dividing the 2 equations):
$$ 2C/C = (A_0 r ^ {t'}) / A_0 r^t $$
Which simplifies to:
$$ 2 = r^{(t' - t)} $$
Solve for $ t' - t $:
$$ t' - t = \frac{\log 2}{\log r} = \frac{\log2}{a}$$
Important note! We haven't specified what the starting time $ t $ actually is, and we don't have to since we just proved that this holds for any $ t $.
"""
print("Time to double:", np.log(2) / a)

# If you know the analytical solution
X = np.array(X).flatten()
Y = np.array(Y)
denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean()*X.sum() ) / denominator
b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator
print(a, b)
print("Time to double:", np.log(2) / a)

"""# Part 2: Making Predictions

This goes with the lecture "Making Predictions"
"""
str2 = '''model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(1,)),
  tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.001,
        momentum=0.9),
        loss='mse')
'''
endtime = time.perf_counter()
duration = round(endtime - starttime,1)


plt.style.use('classic')

# Make sure the line fits our data
Yhat = model.predict(X).flatten()

plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.legend()
plt.title(f'Udemy TensorFlow 2.0 Lecture 14 {modelstart}\n ' +
          f'Linear Regression Moore\'s law {data.data.shape[0]} records ' +
          f'{data.data.shape[1]} fields.  {duration} secds')
# plt.xlabel(f'Udemy TensorFlow 2.0 Duration: {duration}')
plt.text(-39, 16,# transform=trans1,
          s=str2,
          wrap=True, ha='left', va='bottom',
          fontsize=11, bbox=dict(facecolor='yellow', alpha=0.5))
plt.text(-39, 12,# transform=trans1,
         s='Prediction uses learning rate of .001, and momentum=.9 . ' + 
         '\nImplemented with Keras \'dense\' layer' +
         '\nTo train, use the compile() func to specify optimizer an object' +
         '\n loss (mse), and NOT Accuracy',
         wrap=True, ha='left', va='bottom',
         fontsize=11, bbox=dict(facecolor='aqua', alpha=0.5))
plt.text(-39, 6,# transform=trans1,
         s='Activation function ommitted for linear regression because \n' + 
         'the target can be any real number (not just between 0 & 1)' +
         '\nAccuracy not used either...no such concept; in classification\n' +
         'you either have a 0 or 1 (target) and a prediction of 0 or 1.' +
         ',\nbut in regression you have a measurable distance to the line.',
         wrap=True, ha='left', va='bottom',
         fontsize=11, bbox=dict(facecolor='pink', alpha=0.5))
plt.xlabel('tf2_linear_regression.py data: https:/lazyprogrammer/mlexamples/moore.csv')
plt.show()

# Manual calculation

# Get the weights
w, b = model.layers[0].get_weights()

# Reshape X because we flattened it again earlier
X = X.reshape(-1, 1)

# (N x 1) x (1 x 1) + (1) --> (N x 1)
Yhat2 = (X.dot(w) + b).flatten()

# Don't use == for floating points
np.allclose(Yhat, Yhat2)