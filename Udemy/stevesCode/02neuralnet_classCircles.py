# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 08:39:29 2021

@author: steve
Revisit on 15 Aug 2021... apply new 'template' of commenting & timing & chart titling
Also, verify it works with new environment.

INPUT:  Manually create two 'circles' out of points... one circle inside the other.
GOAL: Use an ANN to separate the two circles.
OUTPUT: 7 charts 3rd is very nice combo!

RUNTIME: 38 seconds
"""
# for below, had to install: tensorflow-mkl
# import mkl  I can't get this to load. don't know why.  23 Jun 2021
'''
https://github.com/tensorflow/tensorflow/issues/45853
The TF with MKL (math Kernel Library) needs to be set optimization setting.
I try following setting in linux, got shorter time than stack TF in small_model (CNN).
But a little longer time than stack TF in medium model (LSTM).
Different model need different setting to reach best performance on CPU.
You could try to adjust it based on your CPU.

set TF_ENABLE_MKL_NATIVE_FORMAT=1  
set TF_NUM_INTEROP_THREADS=1
set TF_NUM_INTRAOP_THREADS=4
set  OMP_NUM_THREADS=4
set KMP_BLOCKTIME=1
set KMP_AFFINITY=granularity=fine,compact,1,0
'''
#%% Imports
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
print(tf.__version__)
from sklearn.datasets import make_circles
from datetime import datetime
from sklearn.model_selection import train_test_split
import time

#path = 'C:/Users/steve/Documents/GitHub/DeepLearning/Udemy/tensorflow-deep-learning/stevesCode'
path = 'C:/Users/steve/Documents/GitHub/misc'
sys.path
sys.path.insert(0,path)
import tensorPrepStarter as tps
starttime, modelstart = tps.getstart()

try:
     filename = os.path.basename(__file__)
except NameError:
     filename = 'working'
# !nvidia-smi

'''how to use the function make_circles to make two “circle classes” for your 
    machine learning algorithm to classify. The method takes two inputs: the 
    amount of data you want to generate n_samples and the noise level in the 
    data noise.
'''
#%% Create sample data
n_samples = 1000
X,y = make_circles(n_samples, noise = .03, random_state=42)

X
y[:10]
X[:10]
import pandas as pd
circles = pd.DataFrame({"X0":X[:,0], 'X1':X[:,1], 'label':y})
circles.head()
circles.label.value_counts()

#%% 1st Plot 
import matplotlib.pyplot as plt
plt.title(f'the dots as generated... in {filename}')
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdYlBu)
plt.show()
X.shape, y.shape
len(X), len(y)
circles.head()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#%% split the data so we have something to test... 200 points to be exact.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.2, random_state=42)

print(tf.version.GIT_VERSION, tf.version.VERSION)

from tensorflow.python import _pywrap_util_port
print("MKL enabled:", _pywrap_util_port.IsMklEnabled())

# tf.get_logger().setLevel('INFO')
# tf.get_logger().setLevel('ERROR')  # THIS SEEMED TO WORK! :) 
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
# import logging
# tf.get_logger().setLevel(logging.ERROR)
#%%def plot_decision_boundary(model, title1 , X=X, y=y!
# Lesson 75 
import numpy as np

def plot_decision_boundary(model, title1 , X=X, y=y):
  """
  Plots the decision boundary created by a model predicting on X.
  This function has been adapted from two phenomenal resources:
   1. CS231n - https://cs231n.github.io/neural-networks-case-study/
   2. Made with ML basics - https://github.com/madewithml/basics/blob/master/notebooks/09_Multilayer_Perceptrons/09_TF_Multilayer_Perceptrons.ipynb
  """
  # Define the axis boundaries of the plot and create a meshgrid
  x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))
  
  # Create X values (we're going to predict on all of these)
  # .c_  turns two 'stacked arrays' flat...!
  x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2D arrays together: https://numpy.org/devdocs/reference/generated/numpy.c_.html
  acc = round( model.evaluate(X, y)[1],2)
  # Make predictions using the trained model
  y_pred = model.predict(x_in)

  # Check for multi-class
  if len(y_pred[0]) > 1:
    print("doing multiclass classification...")
    # We have to reshape our predictions to get them ready for plotting
    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
    print("doing binary classifcation...")
    y_pred = np.round(y_pred).reshape(xx.shape)
  
  # Plot decision boundary
  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
  title1 = title1 +'\n Accuracy: ' + str(acc)
  plt.title(title1)
  #plt.show()
#  return y_pred[0]
 

#%% 1st Model
tf.random.set_seed(42)
epochs = 25
#1 Create the model
model1 = tf.keras.Sequential([tf.keras.layers.Dense(1)
 ]                         )
#2 Compile the model
model1.compile(loss=tf.keras.losses.BinaryCrossentropy(),  #For a CLASSIFICATION problem
               optimizer=tf.keras.optimizers.SGD(),
               metrics='accuracy')
#3 Fit the model (train)
model1.fit(X, y, epochs=epochs, verbose=0)
model1.evaluate(X,y)
title2 = f'Model1: One dense layer: 1; Opt: SGD; epochs {epochs}'
plot_decision_boundary(model1, title2, X, y)
# model1.evaluate(X, y)
print(model1.metrics_names)
plt.show()
#%% 2nd Model
tf.random.set_seed(42)
#1 Create the model
model2 = tf.keras.Sequential([tf.keras.layers.Dense(100)
                             ,tf.keras.layers.Dense(1)
 ]                         )
#2 Compile the model
model1.compile(loss=tf.keras.losses.BinaryCrossentropy(),  #For a CLASSIFICATION problem
               optimizer=tf.keras.optimizers.SGD(),
               metrics='accuracy')
#3 Fit the model (train)
model1.fit(X, y, epochs=100, verbose=0)
model1.evaluate(X,y)
title2 = 'Model2: Only Two dense layers: 100, 1; Opt: Adam; ep 100'
plot_decision_boundary(model1, title2, X, y)
plt.show()
#%% 3rd Model  Lesson 74
tf.random.set_seed(42)
#1 Create the model
model2 = tf.keras.Sequential([tf.keras.layers.Dense(100)
                              ,tf.keras.layers.Dense(10)
                             ,tf.keras.layers.Dense(1)
 ]                         )
#2 Compile the model
model1.compile(loss=tf.keras.losses.BinaryCrossentropy(),  #For a CLASSIFICATION problem
               optimizer=tf.keras.optimizers.Adam(),
               metrics='accuracy')
#3 Fit the model (train)
model1.fit(X, y, epochs=100, verbose=0)
model1.evaluate(X,y)
title2 = 'Model3: 3 dense layers: 100,10, 1; Opt: Adam; ep 100'
plot_decision_boundary(model1, title2, X, y)
plt.show()
#%% Set up a new idea... BUT NOT GOING TO GRAPH...CONFUSES THE MAIN TEACHING POINT
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))

#%%  regression problem... We're trying to see if model 3 designed with 
    #circles in mind...will work on different data...
# aGAIN... NOT GOING TO GRAPH BECAUSE IT OBSCURES THE TEACHING POINT  
# Set random seed
tf.random.set_seed(42)

# Create some regression data
X_regression = np.arange(0, 1000, 5)
y_regression = np.arange(100, 1100, 5)

# Split it into training and test sets
X_reg_train = X_regression[:150]
X_reg_test = X_regression[150:]
y_reg_train = y_regression[:150]
y_reg_test = y_regression[150:]
# #%% Run a regression example
# tf.random.set_seed(42)
# model_3 = tf.keras.Sequential([tf.keras.layers.Dense(100),
#                                tf.keras.layers.Dense(10),
#                                tf.keras.layers.Dense(1)]
#     )
# model_3.compile(loss=tf.keras.losses.mae,  # notice change in loss function to be regression specific
#                 optimizer=tf.keras.optimizers.Adam(),
#                 metrics = ['mae'])
# model_3.fit(X_reg_train, y_reg_train, epochs=15)
# # Make predictions with our trained model
# y_reg_preds = model_3.predict(y_reg_test)

# # Plot the model's predictions against our regression data
# plt.figure(figsize=(10, 7))
# plt.scatter(X_reg_train, y_reg_train, c='b', label='Training data')
# plt.scatter(X_reg_test, y_reg_test, c='g', label='Testing data')
# plt.scatter(X_reg_test, y_reg_preds.squeeze(), c='r', label='Predictions')
# plt.title('Model3: (Lesson 76) 3 dense Layers 100, 10,1; Loss=MAE' +
#           '\n This is to see if our ANN model works on different data...')
# plt.legend();
# plt.show()

#%% 4th v1 Model 1 layer w/linear activiation Lesson 77
explanation = 'Changes to find a non-linear boundary":'\
    '\nLayers:   3 DENSE 100,10,1; activation CHANGED to relu' \
    '\nNotice the color is reversed, RED is INSIDE!'\
    '\nBut...BIG help --> CIRCULARITY ACHIEVED (yeay!)\nbut, Accuracy is WORSE!'
tf.random.set_seed(42)
epochs=15
model_4 = tf.keras.Sequential([tf.keras.layers.Dense(100, activation='relu')
                              ,tf.keras.layers.Dense(10, activation='relu')
                             ,tf.keras.layers.Dense(1, activation='relu')
 ]                         )
#    tf.keras.layers.Dense(1, activation='sigmoid')])
model_4.compile(loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(lr=.001), #default
        metrics=['accuracy'])
history = model_4.fit(X, y, epochs=epochs)

plot_decision_boundary(model_4, f'Mdl4 v1: Relu Activation Epochs: {epochs}', X, y)
plt.text(-.51, .4,# transform=trans1,
          s=explanation,
          wrap=True, ha='left', va='top', fontname='Consolas',
          fontsize=7, bbox=dict(facecolor='white', alpha=0.8))
plt.show()
#%% 4th v1a Model 1 layer w/linear activiation Lesson 77
explanationv1a = 'Change to fix "finding the boundary":'\
            '\nHow would we know to drop a layer!!!???'\
    '\nLayers:   AFTER relu: drop a layer (serendipity)' \
        '\nBIG help! Accuracy is GREAT! AND...' \
            '\nBlue switched back to INSIDE, where it should be'
tf.random.set_seed(42)
epochs=15
model_4v1a = tf.keras.Sequential([tf.keras.layers.Dense(100, activation='relu')
#                              ,tf.keras.layers.Dense(10, activation='relu')
                             ,tf.keras.layers.Dense(1, activation='relu')
 ]                         )
#    tf.keras.layers.Dense(1, activation='sigmoid')])
model_4v1a.compile(loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(lr=.001), #default
        metrics=['accuracy'])
history = model_4v1a.fit(X, y, epochs=epochs)

plot_decision_boundary(model_4v1a, f'Mdl4 v1a: with Relu; remove middle layer {epochs}', 
                       X, y)
plt.text(-.51, .4,# transform=trans1,
          s=explanation,
          wrap=True, ha='left', va='top', fontname='Consolas',
          fontsize=7, bbox=dict(facecolor='white', alpha=0.8))
plt.show()
#%% 4th v2 Model 1 layer w/sigmoid Lesson 77 (my idea...switch to sigmoid)
explanationv2 = 'Changes to implement non-linearity:'\
            '\nSigmoid DID NOT HELP! IT HURT!'\
    '\nLayers:      Still 2 (100 neur & 1' \
    '\nNeurons:     Still 100, 1\nActivations: # Changed to SIGMOID' \
        '\ndropped some accuracy (based on Training set)'
epochs=15
tf.random.set_seed(42)
model_4v2 = tf.keras.Sequential([tf.keras.layers.Dense(100, activation='relu')
 #                             ,tf.keras.layers.Dense(10, activation='relu')
                             ,tf.keras.layers.Dense(1, activation='sigmoid')
 ]                         )
model_4v2.compile(loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(lr=.001),
        metrics=['accuracy'])
history = model_4v2.fit(X, y, epochs=epochs)

plot_decision_boundary(model_4v2, 
                       f'Mdl4 v2: 3Dense BinCrossEntropy, SIGMOID activat {epochs}',
                       X, y)
plt.text(-.51, .4,# transform=trans1,
          s=explanation,
          wrap=True, ha='left', va='top', fontname='Consolas',
          fontsize=7, bbox=dict(facecolor='white', alpha=0.8))
plt.show()
#%% 4th v3 Model 1 layer w/sigmoid Lesson 77 (my idea...alter # of layers & LR & Epochs)
explanationv3 = 'Changes to implement non-linearity: playground.tensorflow'\
    '\nAdding a layer, dropping neurons, raising LR to .01!!!...'\
    '\nLayers:      back to 3, but...' \
    '\nNeurons:     4 (drastic drop from 100,10) (1st 2 layers)'\
    '\nActivations: Same (relu,sigmoid)' \
    '\nWITHOUT LR BEING RAISED, ACCURACY WOULD DROP TO .5!!!'
epochs=15
tf.random.set_seed(42)
model_4v3 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation='sigmoid')])
model_4v3.compile(loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(.01),
        metrics=['accuracy'])
history = model_4v3.fit(X, y, epochs=epochs)

plot_decision_boundary(model_4v3, f'Mdl4 v3: Dense 441 LR: .01! Epochs: {epochs}', X, y)
plt.text(-.6, .3,# transform=trans1,
          s=explanation,
          wrap=True, ha='left', va='top', fontname='Consolas',
          fontsize=7, bbox=dict(facecolor='white', alpha=0.8))
plt.show()
#%% Plot all 4 versions of model 4...
plt.figure(figsize=(12,12))
#plt.suptitle(f'Udemy TensorFlowCertZtoM Lect# 73 Using a NN to distinguish 2 cirles {modelstart}' +
plt.suptitle(f'Udemy TensorFlowCertZtoM Lect# 67 thru 80 Using a NN to distinguish 2 cirles Tue, June 29 2021' +
             '\ncodeFile: \\github\\Deeplearning\\02neuralnet_classCircles.py' + 
             '\nINPUT: 1000 dots, 500 red outside cicle, 500 blue inside circle' +
             '\nThe Model creates a "Decision boundary" around the blue...' )
plt.subplot(2,2,1)
plot_decision_boundary(model_4, f'Mdl4 v1: Relu Activation Epochs: {epochs}', X, y)
plt.xticks([]); plt.yticks([])
plt.text(-.51, .4,# transform=trans1,
          s=explanation,
          wrap=True, ha='left', va='top', fontname='Consolas',
          fontsize=7, bbox=dict(facecolor='white', alpha=0.8))

plt.subplot(2,2,2)
plot_decision_boundary(model_4v1a, f'Mdl4 v1a: with Relu; remove middle layer {epochs}', 
                       X, y)
plt.xticks([])
plt.yticks([])
plt.text(-.51, .4,# transform=trans1,
          s=explanationv1a,
          wrap=True, ha='left', va='top', fontname='Consolas',
          fontsize=7, bbox=dict(facecolor='white', alpha=0.8))

plt.subplot(2,2,3)
plot_decision_boundary(model_4v2, 
                       f'Mdl4 v2: 3Dense BinXEntropy, SIGMOID activat {epochs}',
                       X, y)
plt.xticks([])
plt.yticks([])
plt.text(-.51, .4,# transform=trans1,
          s=explanationv2,
          wrap=True, ha='left', va='top', fontname='Consolas',
          fontsize=7, bbox=dict(facecolor='white', alpha=0.8))

plt.subplot(2,2,4)
plot_decision_boundary(model_4v3, f'Mdl4 v3: Dense 441 LR: .01! Epochs: {epochs}', X, y)
plt.xticks([])
plt.yticks([])

plt.text(-.6, .3,# transform=trans1,
          s=explanationv3,
          wrap=True, ha='left', va='top', fontname='Consolas',
          fontsize=7, bbox=dict(facecolor='white', alpha=0.8))
plt.show()
#%% 5th Model from playground.tensorflow.org  Lesson 78
explanation5 = 'Changes to implement non-linearity:\nLayers:      From 1 to 3' \
    '\nNeurons:    1 to 4 (1st two layers) \nActivations: 1st 2 layers Relu' \
    '\n3rd layer    "sigmoid" \nLearning rate increased 10 fold' \
        '\nWhich was most important? All Impacted' \
            '\nAdding epochs could make up for fewer layer(s)'
epochs=19 
tf.random.set_seed(42)
model5 = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
# model5.compile(loss=tf.keras.losses.BinaryCrossentropy(),
model5.compile(loss='binary_crossentropy',
               optimizer=tf.keras.optimizers.Adam(lr=0.01),
               metrics = ['accuracy'])                   
history = model5.fit(X, y, epochs=epochs)
model5.evaluate(X, y)
plot_decision_boundary(model5, 'Model 5: Playground design (4,4,1; BinaryCrossEntropy', X, y)
plt.text(-.51, .4,# transform=trans1,
          s=explanation5,
          wrap=True, ha='left', va='top', fontname='Consolas',
          fontsize=7, bbox=dict(facecolor='white', alpha=0.8))
plt.show()
#%%  Restart 1
n_samples = 1000
X,y = make_circles(n_samples, noise = .03, random_state=42)

len(X)
X, y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.2, random_state=42)

X_train.shape, y_train.shape
#let's recreate model on training...test on test!! 
tf.random.set_seed(42)
model6 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])
model6.compile(loss='binary_crossentropy',
               optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
               metrics= ['accuracy'])
history = model6.fit(X_train, y_train, epochs=24)
model6.evaluate(X_test, y_test)
#%% Document the model...
mdltxt = model6.summary()
stringlist = []
model6.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)
#%% visualize...show 4 charts in one
hist = pd.DataFrame(history.history)
modelstart = time.strftime('%c')
circleinput = '''In [329] circles.head()
Out[329]: 
       X0      X1          label
0  0.7542  0.2381      1
1 -0.7561  0.1559      1
2 -0.8153  0.1782      1
3 -0.3937  0.6983      1
4  0.4422 -0.8923      0'''

code = '''model6 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])
model6.compile(loss='binary_crossentropy',
               optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
               metrics= ['accuracy'])
history = model6.fit(X_train, y_train, epochs=24)
'''

neurons4reason = '''playground.tensorflow.org 'worked' with 2 layers, each
with 4 neurons.  That's where this model came from, and how the 
# of layers and neurons was selected.  serendipity.'''

plt.figure(figsize=(12,12))
plt.suptitle(f'Udemy TensorFlowCertZtoM Lect# 73 Using a NN to distinguish 2 cirles {modelstart}' +
#plt.suptitle(f'Udemy TensorFlowCertZtoM Lect# 73 Using a NN to distinguish 2 cirles Tue, June 29 2021' +
             '\ncodeFile: \\github\\Deeplearning\\02neuralnet_classCircles.py' + 
             '\nINPUT: 1000 dots, 500 red outside cicle, 500 blue inside circle' +
             '\nThe Model creates a "Decision boundary" around the blue...' )
plt.subplot(2,2,1)
plt.title('"created artificial" circle data and model')
plt.text(.1, .6, circleinput)
plt.text(.1, .1,# transform=trans1,
          s=short_model_summary,
          wrap=True, ha='left', va='bottom', fontname='Consolas',
          fontsize=7, bbox=dict(facecolor='pink', alpha=0.5))

plt.subplot(2,2,2)
plt.plot(hist.loss, label='Loss')
plt.plot(hist.accuracy, label='Accuracy')
plt.text(1,.2, s=code, ha='left', va='bottom', fontname='Consolas',
          fontsize=7, bbox=dict(facecolor='blue', alpha=0.2))
plt.text(1,.7, s=neurons4reason, ha='left', va='bottom', fontname='Consolas',
          fontsize=7, bbox=dict(facecolor='green', alpha=0.2))
plt.title('Mdl6 loss curves LR=.01, Loss=BinCRosEnt')
plt.legend()
plt.subplot(2,2,3)
plt.title('Training')
plot_decision_boundary(model6,'Mdl6:Loss=BinCRosEnt, met=Accuracy',  X_train, y_train)
plt.subplot(2,2,4)
plt.title ('test')
plot_decision_boundary(model6,'Model6: on Test Data', X_test, y_test)
plt.show();

# #%% visualize the training history; plot the loss (training curves)
# history.history
# pd.DataFrame(history.history)
# pd.DataFrame(history.history).plot()
# plt.title('Model6 loss curves: LR=.01, Loss=BinCRosEnt, metric=Accuracy')
# plt.show();

#%% lesson 84:  find the ideal learning rate...
# a learning rate callback.  TENSORBOARD!!!
#%%  # Define the Keras TensorBoard callback.
logdir="d:/data/logs/TFcertUdemy/" + datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
              histogram_freq=1,                                     
              profile_batch='500,520')  #this seemed to fix the errors noted in the opening dialog above. :) 
file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')
#%%  # Creates a file writer for the log directory.
imagedir = logdir+'/imgs'
file_writer = tf.summary.create_file_writer(imagedir)
#%% 
tf.random.set_seed(42)
model7 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu')
    ,tf.keras.layers.Dense(4, activation='relu')
    ,tf.keras.layers.Dense(1, activation='sigmoid')])
model7.compile(loss=tf.keras.losses.BinaryCrossentropy(),
               optimizer=tf.keras.optimizers.Adam(learning_rate=.01),
               metrics=['accuracy'])
# create the LR callback...
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10 **(epoch/20))

history7 = model7.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_callback,lr_scheduler], verbose=0)
model7.evaluate(X_train, y_train)
#%%

pd.DataFrame(history7.history).plot(figsize=(10,7), xlabel='epochs')
# Plot the learning rate versus the loss
lrs = 1e-4 * (10 ** (np.arange(100)/20))
lrs
plt.figure(figsize=(10, 7))
plt.semilogx(lrs, history7.history["loss"]) # we want the x-axis (learning rate) to be log scale
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning rate vs. loss");
#%%  Restart 2
# n_samples = 1000
# X,y = make_circles(n_samples, noise = .03, random_state=42)
# create the LR callback...
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10 **(epoch/20))

logdir="d:/data/logs/TFcertUdemy/mdl8_" + datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
              histogram_freq=1,                                     
              profile_batch='500,520')  #this seemed to fix the errors noted in the opening dialog above. :) 
tf.random.set_seed(42)
model8 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])
model8.compile(loss='binary_crossentropy', #  tf.keras.losses.BinaryCrossentropy(),
               optimizer=tf.keras.optimizers.Adam(lr=.02),
               metrics=['accuracy'])
history8 = model8.fit(X_train, y_train, epochs=20 , callbacks=[tensorboard_callback,lr_scheduler], verbose=0)
loss, accuracy = model8.evaluate(X_test, y_test)
print(f'Loss & Accuracy: {loss:.2f}, {accuracy*100:.2f}%')
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')
# Plot the decision boundaries for the training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model8, 'Model8', X=X_train, y=y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model8, 'Model8 test data', X=X_test, y=y_test)
plt.show()
#%%  confusion matrix...
from sklearn.metrics import confusion_matrix
import seaborn as sns
y_pred = model8.predict(X_test) 
confusion_matrix(y_test, tf.round(y_pred))

con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
classes=[0,1]
con_mat_df = pd.DataFrame(confusion_matrix(y_test, tf.round(y_pred)),
                     index = classes, 
                     columns = classes)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues,annot_kws={"size":28})
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#%% ENDING
duration =  time.perf_counter() - starttime
if duration < 60:
    print('it  took: {:.2f} seconds'.format(duration))
elif duration >= 60: print('it  took: {:.2f} minutes'.format(duration/60))
