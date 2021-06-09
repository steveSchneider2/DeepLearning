#%% Import1
# https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/TensorFlow/Basics
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#%% Import Tensorflow
import tensorflow as tf
import tensorflow_estimator as tfe
import tensorflow_datasets as tfds
#%% Get GPU & TF status
print(tf.__version__)
from tensorflow.python.client import device_lib 
# print(device_lib.list_local_devices())  # this puts out a lot of lines (Gibberish?)
print('Conda Envronment:  ', os.environ['CONDA_DEFAULT_ENV'])
print(f'Gpu  Support:       {tf.test.is_built_with_gpu_support()}')
print(f'Cuda Support:       {tf.test.is_built_with_cuda()}')
print(f'Tensor Flow:        {tf.version.VERSION}')
pver = str(format(sys.version_info.major) +'.'+ format(sys.version_info.minor)+'.'+ format(sys.version_info.micro))
print('Python version:      {}.'.format(pver)) 
print('The numpy version:   {}.'.format(np.__version__))
# print('The panda version:   {}.'.format(pd.__version__))
tf.test.gpu_device_name()
tf.test.__package__
tf.test.__doc__
#%%
from tensorflow import keras
from tensorflow.keras import layers
#%%
# Make sure we don't get any GPU errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#%%
# the following worked for cifar100, mnist, titanic.  Not for 4 others.
# (ds_train, ds_test), ds_info = \
#     tfds.load(  'deep_weeds', #  "super_glue",   # "cifar10",    
#         split=["train", "test"],    shuffle_files=True,    
#         data_dir='d:/data/tensorflow/',    as_supervised=True,    
#         with_info=True,)

# ds_minst, ds_info = \
#     tfds.load(    "cifar10",    
#         split=["train", "test"],    shuffle_files=True,    
#         data_dir='d:/data/tensorflow/', #   as_supervised=True,    
#         with_info=True,)

cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()
print("x_train.shape:", x_train.shape)
print("y_train.shape", y_train.shape)



#%%
def normalize_img(image, label):
    """Normalizes images"""
    return tf.cast(image, tf.float32) / 255.0, label


def augment(image, label):
    if tf.random.uniform((), minval=0, maxval=1) < 0.1:
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])

    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_flip_left_right(image)

    return image, label

#%%
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

# Setup for train dataset
x_train = x_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
x_train = x_train.cache()
x_train = x_train.shuffle(ds_info.splits["train"].num_examples)
x_train = x_train.map(augment)
x_train = x_train.batch(BATCH_SIZE)
x_train = x_train.prefetch(AUTOTUNE)

# Setup for test Dataset
ds_test = x_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = x_train.batch(BATCH_SIZE)
ds_test = x_train.prefetch(AUTOTUNE)
#%%
class_names = [
    "Airplane",
    "Autmobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]


def get_model():
    model = keras.Sequential(
        [
            layers.Input((32, 32, 3)),
            layers.Conv2D(8, 3, padding="same", activation="relu"),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(10),
        ]
    )

    return model


model = get_model()

model.compile(
    optimizer=keras.optimizers.Adam(lr=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir="tb_callback_dir", histogram_freq=1,
)
epochs = 5
mdl_cifar10 = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        callbacks=[tensorboard_callback],
                        verbose=2,
                        epochs=epochs)

# model.fit(
#     x_train,
#     epochs=5,
#     validation_data=ds_test,
#     callbacks=[tensorboard_callback],
#     verbose=2,
#)

# %%
