# -*- coding: utf-8 -*-
"""TF2.0 Linear Classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16kgx8sv9v3dunBeMc_Db4YzAUFkziA1_
"""

# Commented out IPython magic to ensure Python compatibility.
# Install TensorFlow
# !pip install -q tensorflow-gpu==2.0.0-beta1

# try:
# #   %tensorflow_version 2.x  # Colab only.
# except Exception:
#   pass
import time
import os
import sys
import numpy as np
import pandas as pd
import pydot
import graphviz
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

starttime = time.perf_counter()
modelstart = time.strftime('%c')

print(tf.__version__)
tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Load in the data

# from tensorflow.python.client import device_lib 
# print(device_lib.list_local_devices())
print('Conda Envronment:  ', os.environ['CONDA_DEFAULT_ENV'])
print(f'Gpu  Support:       {tf.test.is_built_with_gpu_support()}')
print(f'Cuda Support:       {tf.test.is_built_with_cuda()}')
print(f'Tensor Flow:        {tf.version.VERSION}')
pver = str(format(sys.version_info.major) +'.'+ format(sys.version_info.minor)+'.'+ format(sys.version_info.micro))
print('Python version:      {}.'.format(pver)) 
print('The numpy version:   {}.'.format(np.__version__))
print('The panda version:   {}.'.format(pd.__version__))

from sklearn.datasets import load_breast_cancer

# load the data
data = load_breast_cancer()

# check the type of 'data'
type(data)

# note: it is a Bunch object
# this basically acts like a dictionary where you can treat the keys like attributes
data.keys()

# 'data' (the attribute) means the input data
data.data.shape
# it has 569 samples, 30 features

# 'targets'
data.target
# note how the targets are just 0s and 1s
# normally, when you have K targets, they are labeled 0..K-1

# their meaning is not lost
data.target_names

# there are also 569 corresponding targets
data.target.shape

# you can also determine the meaning of each feature
data.feature_names

# normally we would put all of our imports at the top
# but this lets us tell a story
from sklearn.model_selection import train_test_split


# split the data into train and test sets
# this lets us simulate how our model will perform in the future
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)
N, D = X_train.shape

# Scale the data
# you'll learn why scaling is needed in a later course
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Now all the fun Tensorflow stuff
# Build the model
# use 'activation' = 'sigmoid to get the output in range of 0 to 1.
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Input(shape=(D,)),
#   tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# Alternatively, you can do:
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(D,), activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#%% Graph the model
import pydot
import graphviz
import matplotlib.pyplot as plt
from tensorflow import keras
# from tensorflow.keras import layers
keras.utils.plot_model(model, to_file='mdl.png', show_shapes=True, expand_nested=True)
image=plt.imread('mdl.png')
plt.imshow(image)
#%% Train the model
r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)


# Jan 25, 2021: added pydot & Graphviz libraries; also added graphviz to PATH
keras.utils.plot_model(model, show_shapes=True)

# Evaluate the model - evaluate() returns loss and accuracy
print("Train score:", model.evaluate(X_train, y_train))
print("Test score:", model.evaluate(X_test, y_test))


endtime = time.perf_counter()
duration = round(endtime - starttime,3)

# Plot what's returned by model.fit()

str3 = '''model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1, activation='sigmoid')])'''

str2 = 'Features are computed from a digitized image of a fine needle\naspirate (FNA) of a breast mass.  They describe\ncharacteristics of the cell nuclei present in the image.\nNumber of Instances: 569\n    :Number of Attributes: 30 numeric, predictive attributes'
condaenv = os.environ['CONDA_DEFAULT_ENV']

plt.style.use('classic')
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')

# Plot the accuracy too
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend(loc='upper right')
plt.title(f'Udemy TenFlow 2.0 Lecture 12 {modelstart}\n ' +
          f'Linear Classification: Breast Cancer {data.data.shape[0]} records ' +
          f'{data.data.shape[1]} fields')
plt.xlabel(f'ANN: Art Neural Net tf2_12ANNLinearClassifBreastCancer.py   Duration: {duration}')
plt.ylim(0,1)
plt.text(2, .6,# transform=trans1,
         s=str3,
         wrap=True, ha='left', va='bottom',
         fontsize=11, bbox=dict(facecolor='yellow', alpha=0.5))
plt.text(2, .3,# transform=trans1,
         s='Prediction uses the rounded sigmoid to get the output in range of 0 to 1. ' + 
         '\nImplemented with Keras \'dense\' layer' +
         '\nTo train, use the compile() func to specify optimizer (adam)' +
         '\n loss (binary cross entropy), and metrics: Accuracy',
         wrap=True, ha='left', va='bottom',
         fontsize=11, bbox=dict(facecolor='aqua', alpha=0.5))
plt.text(2, 0,# transform=trans1,
         s=str2,
         wrap=True, ha='left', va='bottom',
         fontsize=11, bbox=dict(facecolor='pink', alpha=0.5))
plt.text(70, .7,# transform=trans1,
         s=f'Conda Envr:  {condaenv}\n' +
         f'Gpu  Support:       {tf.test.is_built_with_gpu_support()}\n' +
         f'Cuda Support:       {tf.test.is_built_with_cuda()}\n' +
         f'Tensor Flow:        {tf.version.VERSION}\n'+
         f'Python:             {pver}',
         wrap=True, ha='left', va='top',
         fontsize=10, bbox=dict(facecolor='pink', alpha=0.5))
plt.show()
"""# Part 2: Making Predictions

This goes with the lecture "Making Predictions"
"""

# Make predictions passs in a 2-d array for X...
P = model.predict(X_test)
print(P) # they are outputs of the sigmoid, interpreted as probabilities p(y = 1 | x)

# Round to get the actual predictions
# Note: has to be flattened since the targets are size (N,) while the predictions are size (N,1)
import numpy as np
P = np.round(P).flatten()
print(P)

# Calculate the accuracy, compare it to evaluate() output
print("Manually calculated accuracy:", np.mean(P == y_test))
print("Evaluate output:", model.evaluate(X_test, y_test))

"""# Part 3: Saving and Loading a Model

This goes with the lecture "Saving and Loading a Model"
"""

# Let's now save our model to a file
model.save('linearclassifier.h5')

# Check that the model file exists
# !ls -lh

# Let's load the model and confirm that it still works
# Note: there is a bug in Keras where load/save only works if you DON'T use the Input() layer explicitly
# So, make sure you define the model with ONLY Dense(1, input_shape=(D,))
# At least, until the bug is fixed
# https://github.com/keras-team/keras/issues/10417

# model = tf.keras.models.load_model('linearclassifier.h5')
# print(model.layers)
model.evaluate(X_test, y_test)

# Download the file - requires Chrome (at this point)
#from google.colab import files
# files.download('linearclassifier.h5')