# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 20:05:49 2021

@author: steve
"""

# A Sample class with init method   
import tensorflow as tf
class Person:   
      
    # init method or constructor    
    def __init__(self, name):   
        self.name = name   
      
    # Sample Method    
    def say_hi(self):   
        print('Hello, my name is', self.name)   
      
p = Person('Nikhil')   
p.say_hi()   
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Avaialbe: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
