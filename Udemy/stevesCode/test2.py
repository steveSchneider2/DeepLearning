# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 19:48:40 2021

@author: steve
"""
import time , sys, tensorflow as tf
import random, tensorboard

from tensorflow import keras

# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
# logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))

# !nvidia-smi
tensorboard_callback = keras.callbacks.TensorBoard(log_dir='d:/',
              histogram_freq=1,                                     
              profile_batch='500,520')  #this seemed to fix the errors noted in the opening dialog above. :) 
