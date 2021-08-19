# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 10:31:29 2021

@author: steve
"""

import tensorflow as tf
print(tf.__version__)
#%%  this next line is purely experimental. Not sure it has any current value.
tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count = {'GPU': 1}))

#%%
import numpy as np
import matplotlib.pyplot as plt

# Create features
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Create labels
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Visualize it
plt.scatter(X, y);
#%%
house_info = tf.constant(["bedroom", "bathroom", "garage"])
house_price = tf.constant([939700])
house_info, house_price

house_info.shape

# Create features (using tensors)
X = tf.constant([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Create labels (using tensors)
y = tf.constant([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Visualize it
plt.scatter(X, y);

X
X.ndim
X.shape
X[0]
X[0].shape
X[0].ndim
#%%
# Set random seed
tf.random.set_seed(42)

# Create a model using the Sequential API
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1) #,   tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
              optimizer=tf.keras.optimizers.SGD(), # SGD is short for stochastic gradient descent
              metrics=["mae"])

#%%  Fit the model
model.fit(X, y, epochs=100)
model.predict([17])
#%% new data 

X = np.arange(-100, 100, 4)
X
y = np.arange(-90, 110, 4)
y

X_train= X[:40]
X_test = X[40:]
y_train = y[:40]
y_test = y[40:]
len(X_train), len(X_test)
#%%
plt.figure(figsize=(10,7))
plt.scatter(X_train, y_train, c='b', label='Training')
plt.scatter(X_test, y_test, c='g', label ='test')
plt.legend();
plt.show()
#%%
model.summary()
#%% define, create model
# Set random seed
tf.random.set_seed(42)

# Create a model using the Sequential API
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1, input_shape=[1]) #,   tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
              optimizer=tf.keras.optimizers.SGD(), # SGD is short for stochastic gradient descent
              metrics=["mae"])
#%%
model.fit(X_train, y_train, 100)
#%% 
model.summary()
#%%
from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True)

#%%
y_preds = model.predict(X_test)
y_preds
X_test
#%%
def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=y_preds):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))
  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", label="Training data")
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", label="Testing data")
  # Plot the predictions in red (predictions were made on the test data)
  plt.scatter(test_data, predictions, c="y", label="Predictions")
  # Show the legend
  plt.legend()

plot_predictions(train_data=X_train,
                 train_labels=y_train,
                 test_data=X_test,
                 test_labels=y_test,
                 predictions=y_preds)
