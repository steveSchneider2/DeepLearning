# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:06:12 2021
Most important takeaways:... 
    1. image_dataset_from_directory: Faster, simpler
    2. Functional API
Re-examined on Aug 6th... after focusing on object detection & getting my 'history' organized.
@author: steve
Aug  8: E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1661] function cupti_interface_->Subscribe( &subscriber_, (CUpti_CallbackFunc)ApiCallback, this)failed with error CUPTI could not be loaded or symbol could not be found.
TOTAL RUN TIME... 20 to 34 MIN!!! Depends on GPU memory status?

INPUT:  10,000 food images, EfficientNetB4 model.  
OUTPUT: 3 charts comparing Exp 1 to 3, 3 to 4 and exp 4 to Exp 5.
Extra: Uses TensorBoard. (callbacks)

FIRST RUN: Mon Jul 26 14:03:25 2021       

ENVIRONMENT:
    starting at:  Wed Aug 11 14:05:05 2021
Conda Envronment:   MLFlowProtoBuf
Gpu  Support:       True
Cuda Support:       True
Tensor Flow:        2.5.0
Python version:      3.8.8.
The numpy version:   1.19.5.
The panda version:   1.2.4.
Tensorboard version  2.6.0.
"""
# %% Imports...
import numpy as np, matplotlib.pyplot as plt
import os, pandas as pd, seaborn as sns
# 1 July 2021... next two statements...BEFORE any tensorflow did the trick.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time , sys, tensorflow as tf, sklearn.metrics, itertools, io

import random, tensorboard

from datetime import datetime
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.layers.experimental import preprocessing

# import pydot
import graphviz  #for graphs; just python-graphviz  'graphviz' was not enough (what's the difference?) 
import GPUtil
gpus = GPUtil.getGPUs()
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', None)
#path = 'C:/Users/steve/Documents/GitHub/DeepLearning/Udemy/tensorflow-deep-learning/stevesCode'
path = 'C:/Users/steve/Documents/GitHub/misc'

sys.path
sys.path.insert(0,path)
import tensorPrepStarter as tps

# !nvidia-smi

# %% get data
# url = url = 'https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_1_percent.zip'
# zipfilepath = 'd:/data/udemy/dbourkeTFcert'
# tps.unzip_data(zipfilepath, url) 
# %%  # Define the Keras TensorBoard callback.
topdirname = 'D:\Data\logs\TFcertUdemy\\05food10clsTransLearn2\\'
tbcb, logdir = tps.create_tb_callback\
    (topdirname, 'effnetB4', '10perc_noAug')
# %% tps.ShowImageFolder_SetTrainTestDirs
imagefolder = 'd:/data/udemy/dbourkeTFcert/10_food_classes_10_percent'
#imagefolder = 'd:/data/udemy/dbourkeTFcert/10_food_classes_1_percent'

train_dir, test_dir = tps.ShowImageFolder_SetTrainTestDirs(imagefolder)
#%% define:  checkpoints
checkpoint_path = 'D:/Data/udemy/dbourkeTFcert/ten_percent_model_checkpoints_weights/checkpoint.ckpt' # note: remember saving directly to Colab is temporary
checkpoint_callback = tf.keras.callbacks.\
    ModelCheckpoint(filepath=checkpoint_path,
                    save_weights_only=True, # set to False to save the entire model
                    save_best_only=True, # set to True to save only the best model instead of a model every epoch 
                    save_freq="epoch", # save every epoch
                    verbose=1)

# %% Create the data loaders
# normalize 0 to 255; turn data into "Flows of batches of data"

from tensorflow.keras.preprocessing.image import ImageDataGenerator
IMAGE_SHAPE= (224, 224)  # HYPER PARAMETER
BATCH_SIZE = 32

# the following "Image datagen INSTANCE" is not needed in the new function below...
# train_datagen = ImageDataGenerator(rescale=1/255.)
# test_dategen  = ImageDataGenerator(rescale=1/255.)

# below, we are using "image_dataset_from_directory" ...this is faster, simpler (no datagen instance needed)
train_batch = tf.keras.preprocessing.\
    image_dataset_from_directory(train_dir, 
                                 image_size=IMAGE_SHAPE,
                                 batch_size=BATCH_SIZE, seed = 42,
                                 label_mode='categorical') #this is the default, so i don't really need it
test_batch = tf.keras.preprocessing.\
    image_dataset_from_directory(directory=test_dir, image_size=IMAGE_SHAPE,
                                 label_mode='categorical', batch_size=BATCH_SIZE
                                 ,seed=42)
#                BATCH!!                     IMAGES         LABELS, ONE-HOT ENCODED
#train_batch = <BatchDataset shapes: ((None, 224, 224, 3), (None, 10)), types: (tf.float32, tf.float32)>
# notice you do lots of thing (methods)
# Let's look at what we've made with the 'train_batches'...
train_batch.class_names
# below is optional...just if you want to see what the data looks like...
# for images, labels in train_batch.take(1):
#     print(images, labels)
#%% Setup input shape
input_shape = (224, 224, 3)
# Create a frozen base model (also called the backbone)
#%% EXPERIMENT #1. build & FIT (~2 min) (75 IMAGES per class)the model from tf.keras.applications (not from some URL as in 04transfer learning...)
effnetB4_base = tf.keras.applications.efficientnet.\
    EfficientNetB4(include_top=False)
# effnetB4_base = tf.keras.applications.resnet50.ResNet50()
# the model trained on imagenet with 1000 classes, has 1000 outputs
# , weights='imagenet', input_tensor=None,
#     input_shape=None, pooling=None, classes=10,
#     classifier_activation='softmax'
# 2. Freeze the base model (so the pre-learned patterns remain)
effnetB4_base.trainable = False

# 3. Create inputs into the base model...what should the model expect as input?
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")

# 4. If using models like ResNet50V2, add this to speed up convergence, 
#    remove for EfficientNet 
# x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)

# 5. Pass the inputs to the effnetB4_base (note: using tf.keras.applications, 
     # EfficientNet inputs don't have to be normalized)
x = effnetB4_base(inputs)
# Check data shape after passing it to effnetB4_base
print(f"Shape after effnetB4_base: {x.shape}")  # (none, 7, 7, 1280)

# 6. Average pool the outputs of the base model (aggregate all the most important information, reduce number of computations)
x = tf.keras.layers.GlobalAveragePooling2D\
    (name="global_average_pooling_layer")(x)    # (none, 1280)
    
print(f"After GlobalAveragePooling2D(): {x.shape}")
print('EXPERIMENT #1: Following 5 epochs... 45 sec for 1st, 24 for each remaining.')
# 7. Create the output activation layer
outputs = tf.keras.layers.Dense\
    (10, activation="softmax", name="output_layer")(x)

# 8. Combine the inputs with the outputs into a (new) model
effnetB4_0 = tf.keras.Model(inputs, outputs, name='effnetB4_base')

# 9. Compile the model
effnetB4_0.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# 10. Fit the model (we use less steps for validation so it's faster)
modelstart = time.strftime('%c')
starttime = time.perf_counter()
effnetB4hist = effnetB4_0.fit(train_batch,
                                 epochs=5,
                                 steps_per_epoch=len(train_batch),
                                 validation_data=test_batch,
                                 # Go through less of the validation data so epochs are faster (we want faster experiments!)
                                 validation_steps=int(0.25 * len(test_batch)), 
                                 # Track our model's training logs for visualization later
                                 callbacks=[tbcb])
endtime = time.perf_counter()
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')
effnetB4_0.evaluate(test_batch)
# %% Prepare 1st model df's for the chart
# record time data
effnetB4 = round(endtime - starttime,2)
effnetB4df = pd.DataFrame(effnetB4hist.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
effnetB4df.rename(columns = {'index':'epochs'}, inplace=True)
effnetB4df

# effnetB4 = effnetB4_base.summary()
stringlist2 = []
effnetB4_0.summary()
effnetB4_0.summary(print_fn=lambda x: stringlist2.append(x))
effnetB4mdlsum = "\n".join(stringlist2)
effnetB4mdlsum = effnetB4mdlsum.replace('_________________________________________________________________\n', '')
# Check layers in our base model ... prints 236 lines...one per layer
# for layer_number, layer in enumerate(effnetB4_base.layers):
#   print(layer_number, layer.name)
  
#effnetB4_base.summary() # 1280 lines? A lot, anyway, fills teh console
#%% Description of process...no code here
'''
Running a series of transfer learning experiments
We've seen the incredible results of transfer learning on 10% of the training data, what about 1% of the training data?

What kind of results do you think we can get using 100x less data than the original CNN models we built ourselves?

Why don't we answer that question while running the following modelling experiments:
model_0: 10% data
model_1: Use feature extraction transfer learning on 1% of the training data with data augmentation.
model_2: Use feature extraction transfer learning on 10% of the training data with data augmentation.
model_3: Use fine-tuning transfer learning on 10% of the training data with data augmentation.
model_4: Use fine-tuning transfer learning on 100% of the training data with data augmentation.
'''
#%% Experiment #2...  1% of training data ...  7 images per... NOT DOING THIS AGAIN.
tbcb1, logdir = tps.create_tb_callback\
    (topdirname, 'effnetB4', '01perc_DataAug')
imagefolder = 'd:/data/udemy/dbourkeTFcert/10_food_classes_1_percent'
train_dir, test_dir = tps.ShowImageFolder_SetTrainTestDirs(imagefolder)

train_batch = tf.keras.preprocessing.\
    image_dataset_from_directory(train_dir, 
                                 image_size=IMAGE_SHAPE,
                                 batch_size=BATCH_SIZE, seed = 42,
                                 label_mode='categorical') #this is the default, so i don't really need it
#%% Exp #2 Build a data augmentation layer with picture displaying difference
data_augmentation = Sequential([
    preprocessing.RandomFlip('horizontal'),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomRotation(0.2),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomWidth(0.2),
    preprocessing.RandomRotation(10) # 10 degrees
    ], name='data_augmentation')
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.suptitle(f'Compare two images {modelstart}')
plt.title ('before')
image = tps.view_random_image(train_dir, random.choice(train_batch.class_names))
plt.subplot(1,2,2)
plt.title('after')
plt.imshow((tf.squeeze(data_augmentation(tf.expand_dims(image, axis=0))))/255.)
plt.axis('off')
plt.show()
#%% Exp #2 NOT DOING THIS build 2nd model from tf.keras.applications (not from some URL as in 04transfer learning...)
# This shows that training on 7 images per class ... is not enough training data!
#####   NOT GOING TO DO THIS MODEL AGAIN...I HAVE LEARNED THE LESSON!  #####
#####    7 IMAGES TRAINGING IS NOT ENOUGH!!! OK.    
'''
basemdl1 = tf.keras.applications.efficientnet.\
    EfficientNetB4(include_top=False)
basemdl1.trainable = False
#create 'inputs'... which start out independent
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
x = data_augmentation(inputs)
#below, we connect the 'inputs' to the effnetB4_base
x = basemdl1(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D\
    (name="global_average_pooling_layer")(x)    # (none, 1280)
outputs = tf.keras.layers.Dense\
    (10, activation="softmax", name="output_layer")(x)
# 8. Combine the inputs with the outputs into a (new) model
effnetB4_1 = tf.keras.Model(inputs, outputs, name='effnetB4_1_dataaug')
effnetB4_1.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])
modelstart1 = time.strftime('%c')
starttime1 = time.perf_counter()
print('EXPERIMENT #2: Following 5 epochs... 33s, 15s, 90s, 114, .')

effnetB4_1hist = effnetB4_1.fit(train_batch,
                                 epochs=5,
                                 steps_per_epoch=len(train_batch),
                                 validation_data=test_batch,
                                 # Go through less of the validation data so epochs are faster (we want faster experiments!)
                                 validation_steps=int(0.25 * len(test_batch)), 
                                 # Track our model's training logs for visualization later
                                 callbacks=[tbcb1])
endtime1 = time.perf_counter()
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')
#%% Exp #2 Prepare 2nd model df's for the chart
# record time data
effnetB4_1dur = round(endtime1 - starttime1,2)
effnetB4_1df = pd.DataFrame(effnetB4_1hist.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
effnetB4_1df.rename(columns = {'index':'epochs'}, inplace=True)
effnetB4_1df

# effnetB4_1 = effnetB4_1_base.summary()
stringlist1 = []
effnetB4_1.summary()
effnetB4_1.summary(print_fn=lambda x: stringlist1.append(x))
effnetB4_1mdlsum = "\n".join(stringlist1)
effnetB4_1mdlsum = effnetB4_1mdlsum.replace('_________________________________________________________________\n', '')
'''
#%% Experiment #3 10% tccallback & DATA (customize)...
tbcb, logdir = tps.create_tb_callback\
    (topdirname, 'effnetB4', '10perc_DataAug')
imagefolder = 'd:/data/udemy/dbourkeTFcert/10_food_classes_10_percent'
train_dir, test_dir = tps.ShowImageFolder_SetTrainTestDirs(imagefolder)

train_batch = tf.keras.preprocessing.\
    image_dataset_from_directory(train_dir, 
                                 image_size=IMAGE_SHAPE,
                                 batch_size=BATCH_SIZE, seed = 42,
                                 label_mode='categorical') #this is the default, so i don't really need it

#%% Exp #3 build 3rd model from tf.keras.applications (not from some URL as in 04transfer learning...)
# This shows that training on 7 images per class ... is not enough training data!
basemdl2 = tf.keras.applications.efficientnet.\
    EfficientNetB4(include_top=False)
basemdl2.trainable = False
#create 'inputs'... which start out independent
inputs = layers.Input(shape=(224, 224, 3), name="input_layer")
x = data_augmentation(inputs)
#below, we connect the 'augmented inputs' (via the 'x' to the effnetB4_base
x = basemdl2(x, training=False)
x = layers.GlobalAveragePooling2D\
    (name="global_average_pooling_layer")(x)    # (none, 1280)
outputs = layers.Dense\
    (10, activation="softmax", name="output_layer")(x)
# 8. Combine the inputs with the outputs into a (new) model
effnetB4_2 = tf.keras.Model(inputs, outputs, name='effnetB4_2_dataaug')

effnetB4_2.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])
modelstart2 = time.strftime('%c')
starttime3 = time.perf_counter()
print('EXPERIMENT #3: Following 5 epochs... 61S, 40s, 35s 36s, ~30s.')

effnetb2hist = effnetB4_2.fit(train_batch,
                                 epochs=5,
                                 steps_per_epoch=len(train_batch),
                                 validation_data=test_batch,
                                 # Go through less of the validation data so epochs are faster (we want faster experiments!)
                                 validation_steps=int(0.25 * len(test_batch)), 
                                 # Track our model's training logs for visualization later
                                 callbacks=[tbcb, checkpoint_callback])
endtime3 = time.perf_counter()
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')
#%% Exp #3 Prepare 2nd model df's for the chart
# record time data
effnetB4_2dur = round(endtime3 - starttime3,2)
effnetB4_2df = pd.DataFrame(effnetb2hist.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
effnetB4_2df.rename(columns = {'index':'epochs'}, inplace=True)
effnetB4_2df

# effnetB4_2 = effnetB4_2_base.summary()
stringlist1 = []
effnetB4_2.summary()
effnetB4_2.summary(print_fn=lambda x: stringlist1.append(x))
effnetB4_2mdlsum = "\n".join(stringlist1)
effnetB4_2mdlsum = effnetB4_2mdlsum.replace('_________________________________________________________________\n', '')
#%% Exp #3 graph differences 
supttl = 'Udemy TFCertifyZtoM 10 Food Classification TransLearn EfficientNetB4 challenge'
lftTtl = 'CNN No data augmentation(1st run) '
rhtTtl = 'EffnetB4 w/ data augmentation Exp3 '
augmnt = '''From "TensorFlowHub", the augmented model is less accurate!!!'''

# tps.ChartMnistNetworkChanges(dfLeft, dfRight, mdlsummary, dfLtime, dfRtime, rmdlsumry, changes, supttl, ttl1, ttl2)
tps.ChartMnistNetworkChanges(effnetB4df, effnetB4_2df, effnetB4mdlsum, 
                             effnetB4, effnetB4_2dur, effnetB4_2mdlsum, 
                             augmnt, supttl, lftTtl, rhtTtl)
#%% Experiment #4 FINETUNING (customize)...
tbcb, logdir = tps.create_tb_callback\
    (topdirname, 'effnetB4', '10perc_DataAug_fintun10')
imagefolder = 'd:/data/udemy/dbourkeTFcert/10_food_classes_10_percent'
train_dir, test_dir = tps.ShowImageFolder_SetTrainTestDirs(imagefolder)

train_batch = tf.keras.preprocessing.\
    image_dataset_from_directory(train_dir, 
                                 image_size=IMAGE_SHAPE,
                                 batch_size=BATCH_SIZE, seed = 42,
                                 label_mode='categorical') #this is the default, so i don't really need it
checkpoint_path = 'D:/Data/udemy/dbourkeTFcert/ten_percent_finetune_checkpoints_weights/checkpoint.ckpt' # note: remember saving directly to Colab is temporary
checkpoint_callback = tf.keras.callbacks.\
    ModelCheckpoint(filepath=checkpoint_path,
                    save_weights_only=True, # set to False to save the entire model
                    save_best_only=True, # set to True to save only the best model instead of a model every epoch 
                    save_freq="epoch", # save every epoch
                    verbose=1)
    
#%% Exp #4 FINE TUNING ie train the last 6 Epochs Unfreeze & FIT...
# To begin fine-tuning, we'll unfreeze the entire base model by setting its 
# trainable attribute to True. Then we'll refreeze every layer in the base 
# model except for the last 10 by looping through them and setting their 
# trainable attribute to False. Finally, we'll recompile the model.

basemdl2.trainable = True

for layer in basemdl2.layers[:-10]:
    layer.trainable = False
    
effnetB4_2.compile(loss='categorical_crossentropy',
                   optimizer=Adam(lr=0.0001), # lr is 10x lower than before for fine-tuning
                   metrics='accuracy')
# comment next line because they print out over 473 lines...
# for layer_number, layer in enumerate(basemdl2.layers):
#     print(layer_number, layer.name, layer.trainable)
print('Exp 4: Trainable layers in the effnetB4 layer:')
for layer in effnetB4_2.layers:
    print(layer.name, layer.trainable)
print('Exp 4: Trainable layers in the base layer:')
for layer in basemdl2.layers:
    if layer.trainable == True:  print (layer.name)
# Fine tune for another 5 epochs...
fine_tune_epochs = 5 + 5
# Refit the model...
print('EXPERIMENT #4: (Re-training) the last 6 epochs... 60s, 37s, 34s, 35s, 36s, 37.')
starttime4 = time.perf_counter()
effnetb2histft = effnetB4_2.fit(train_batch, epochs=fine_tune_epochs,
                                validation_data=test_batch,
                                initial_epoch=effnetb2hist.epoch[-1], # start from previous last epoch
                                validation_steps=int(0.25 * len(test_batch)),
                                callbacks=[tbcb, checkpoint_callback])
endtime4 = time.perf_counter()
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')
#%% Exp #4 Prepare Fine tuned model df's for the chart
# record time data
effnetB4_3ftdur = round(endtime4 - starttime4,2)
effnetB4_3df = pd.DataFrame(effnetb2histft.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
effnetB4_3df.rename(columns = {'index':'epochs'}, inplace=True)
effnetB4_3df

effnetB4_3 = effnetB4_2.summary()
stringlist1 = []
effnetB4_2.summary(print_fn=lambda x: stringlist1.append(x))
effnetB4_3mdlsum = "\n".join(stringlist1)
effnetB4_3mdlsum = effnetB4_3mdlsum.replace('_________________________________________________________________\n', '')
#%% Exp #4 graph differences 
supttl = 'Udemy TFCertifyZtoM 10 Food Classification TransLearn Efficient Net Lect 166'
lftTtl = rhtTtl # 'CNN EffnetB4 with data augmentation'
rhtTtl = 'EffnetB4 w/data aug & fine tuned Exp4'
augmnt = '''Fine tuning takes up right where the 1st quit
adding 6 more epochs of training at LR = .0001, 
training the last 10 layers of the base effnetB4 model'''

# tps.ChartMnistNetworkChanges(dfLeft, dfRight, mdlsummary, dfLtime, dfRtime, rmdlsumry, changes, supttl, ttl1, ttl2)
tps.ChartMnistNetworkChanges(effnetB4_2df,effnetB4_3df, effnetB4_2mdlsum, 
                             effnetB4_2dur, effnetB4_3ftdur, effnetB4_3mdlsum, 
                             augmnt, supttl, lftTtl, rhtTtl)
#%% Experiment #5 FINETUNE on ALL DATA (customize)...
# we're going to have a model 'grossly' trained on 10%, finetuned on ALL data.
tbcb, logdir = tps.create_tb_callback\
    (topdirname, 'effnetB4', 'Alldata_DataAug_fintun10layers')
imagefolder = 'd:/data/udemy/dbourkeTFcert/10_food_classes_all_data'
train_dir, test_dir = tps.ShowImageFolder_SetTrainTestDirs(imagefolder)

train_batch = tf.keras.preprocessing.\
    image_dataset_from_directory(train_dir, 
                                 image_size=IMAGE_SHAPE,
                                 batch_size=BATCH_SIZE, seed = 42,
                                 label_mode='categorical') #this is the default, so i don't really need it
checkpoint_path = 'D:/Data/udemy/dbourkeTFcert/Alldata_finetune_checkpoints_weights/checkpoint.ckpt' # note: remember saving directly to Colab is temporary
checkpoint_callback = tf.keras.callbacks.\
    ModelCheckpoint(filepath=checkpoint_path,
                    save_weights_only=True, # set to False to save the entire model
                    save_best_only=True, # set to True to save only the best model instead of a model every epoch 
                    save_freq="epoch", # save every epoch
                    verbose=1)

#%% EXP 5: restore 'model 2'
checkpoint_path = 'D:/Data/udemy/dbourkeTFcert/ten_percent_model_checkpoints_weights/checkpoint.ckpt' # note: remember saving directly to Colab is temporary
effnetB4_2.load_weights(checkpoint_path)  # this is a change to the model... therefore will need to recompile.
effnetB4_2.evaluate(test_batch)
for layer in effnetB4_2.layers:
    print(layer.name, layer.trainable)
for layer in basemdl2.layers:
    if layer.trainable == True:  print (layer.name)

effnetB4_2full = effnetB4_2
#%% Compile & fit
effnetB4_2full.compile(loss='categorical_crossentropy', 
                       optimizer=Adam(lr=0.0001),
                       metrics='accuracy')
print('EXPERIMENT #5: (Re-trainingALLdata) the last 6 epochs... 248s, 227s, 222s, 202s, 193s, 189.')
starttime5 = time.perf_counter()

effnetb2histftAll = effnetB4_2full.fit(
    train_batch, epochs=fine_tune_epochs,
    validation_data=test_batch,
    initial_epoch=effnetb2hist.epoch[-1], # start from previous last epoch
    validation_steps=int(0.25 * len(test_batch)),
    callbacks=[tbcb, checkpoint_callback])
endtime5 = time.perf_counter()
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')
#%% Exp #5 Prepare Fine tuned model df's for the chart
# record time data
effnetB4_5dur = round(endtime5 - starttime5,2)
effnetB4_5df = pd.DataFrame(effnetb2histftAll.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
effnetB4_5df.rename(columns = {'index':'epochs'}, inplace=True)
effnetB4_5df

effnetB4_5 = effnetB4_2.summary()
stringlist1 = []
effnetB4_2.summary(print_fn=lambda x: stringlist1.append(x))
effnetB4_5mdlsum = "\n".join(stringlist1)
effnetB4_5mdlsum = effnetB4_5mdlsum.replace('_________________________________________________________________\n', '')
#%% Exp #5 graph differences 
supttl = 'Udemy TFCertifyZtoM 10 Food Classification TransLearn Efficient Net Lect 166'
lftTtl = 'Exp4 EffnetB4 w/data aug & fine tuned'
rhtTtl = 'Exp5 EffnetB4 w/ALLdata aug & fine tuned'
augmnt = '''KEY LEARNING: MORE DATA, MORE TIME
Fine tuning takes up right where the 1st quit
adding 6 more epochs of training at LR = .0001, 
training the last 10 layers of the base effnetB4 model
ALL DATA!!!'''

# tps.ChartMnistNetworkChanges(dfLeft, dfRight, mdlsummary, 
#                              dfLtime, dfRtime, rmdlsumry, changes, supttl, ttl1, ttl2)
tps.ChartMnistNetworkChanges(effnetB4_3df,effnetB4_5df, effnetB4_2mdlsum, 
                             effnetB4_3ftdur, effnetB4_5dur, effnetB4_5mdlsum, 
                             augmnt, supttl, lftTtl, rhtTtl)
#%% ENDING
duration =  time.perf_counter() - starttime
if duration < 60:
    print('it  took: {:.2f} seconds'.format(duration))
elif duration >= 60: print('it  took: {:.2f} minutes'.format(duration/60))
