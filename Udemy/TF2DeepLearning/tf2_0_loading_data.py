# -*- coding: utf-8 -*-
"""TF2.0 Loading Data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mahtZt0sy4rt-HqGdCgEFooC_DwDSTYf

# Part 1: Using wget
"""

# !ipython nbconvert -- to script

# download the data from a URL
# source:                     https://archive.ics.uci.edu/ml/datasets/Arrhythmia
# alternate URL: https://lazyprogrammer.me/course_files/arrhythmia.data
#!wget --no-check-certificate https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data
!wget https://lazyprogrammer.me/course_files/arrhythmia.data

# check the data
import pandas as pd
df = pd.read_csv('data/arrhythmia.data', header=None)
df.head()
# since the data has many columns, take just the first few and name them (as per the documentation)
data = df[[0,1,2,3,4,5]]
data.columns = ['age', 'sex', 'height', 'weight', 'QRS duration', 'P-R interval']

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 15] # make the plot bigger so the subplots don't overlap
data.hist(); # use a semicolon to supress return value

from pandas.plotting import scatter_matrix
scatter_matrix(data);

"""# Part 2: Using tf.keras"""

# use keras get_file to download the auto MPG dataset
# source: https://archive.ics.uci.edu/ml/datasets/Auto+MPG
#url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'

### alternate URL
url = 'https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/auto-mpg.data'

# Install TensorFlow

import tensorflow as tf
print(tf.__version__)

# check out the documentation for other arguments
tf.keras.utils.get_file('auto-mpg.data', url)

# head('data/udemy/auto-mpg.data')

# unless you specify an alternative path, the data will go into /root/.keras/datasets/
# df = pd.read_csv('/root/.keras/datasets/auto-mpg.data', header=None, delim_whitespace=True)
df = pd.read_csv('data/udemy/auto-mpg.data', header=None, delim_whitespace=True)
df.head()

"""# Part 3: Upload the file yourself"""

# another method: upload your own file

##### PLEASE NOTE: IT DOES NOT MATTER WHICH FILE YOU UPLOAD
##### YOU CAN UPLOAD ANY FILE YOU WANT
##### IN FACT, YOU ARE ENCOURAGED TO EXPLORE ON YOUR OWN

# if you must, then get the file from here:
# https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/daily-minimum-temperatures-in-me.csv

from google.colab import files
uploaded = files.upload()

uploaded

# file is uploaded to the current directory
!ls

# open the file
# the last few lines are junk
df = pd.read_csv('daily-minimum-temperatures-in-me.csv', error_bad_lines=False)
df.head()

# upload a Python file with some useful functions (meant for fake_util.py)
from google.colab import files
uploaded = files.upload()

from fake_util import my_useful_function
my_useful_function()

!pwd

"""# Part 4: Access files from Google Drive"""

# Access files from your Google Drive
from google.colab import drive
drive.mount('/content/gdrive')

# Check current directory - now gdrive is there
!ls

# What's in gdrive?
!ls gdrive

# Whoa! Look at all this great VIP content!
!ls '/content/gdrive/My Drive/'