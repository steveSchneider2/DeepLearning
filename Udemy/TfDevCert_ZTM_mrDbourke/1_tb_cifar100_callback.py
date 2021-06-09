#%% 
import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#%%
import tensorflow_datasets as tfds
#%%

from tensorflow import keras
#%%
from tensorflow.keras import layers
#%%
# Make sure we don't get any GPU errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#%%
(ds_train, ds_test), ds_info = \
    tfds.load(    "cifar100",   # "cifar10",    
        split=["train", "test"],    shuffle_files=True,    
        data_dir='d:/data/tensorflow/',    as_supervised=True,    
        with_info=True,)
# ds_minst, ds_info = \
#     tfds.load(    "cifar10",    
#         split=["train", "test"],    shuffle_files=True,    
#         data_dir='d:/data/tensorflow/', #   as_supervised=True,    
#         with_info=True,)
# mnist = tf.keras.datasets.cifar10
# (ds_train, ds_train),(ds_test, ds_test) = mnist.load_data()
# ds_train = pd.DataFrame(ds_train)

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
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.map(augment)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Setup for test Dataset
ds_test = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_train.batch(BATCH_SIZE)
ds_test = ds_train.prefetch(AUTOTUNE)
#%%
class_names = [
    'beaver'
    ,'dolphin'
    ,'otter'
    ,'seal'
    ,'whale'
    ,'aquarium fish'
    ,'flatfish'
    ,'ray'
    ,'shark'
    ,'trout'
    ,'orchids'
    ,'poppies'
    ,'roses'
    ,'sunflowers'
    ,'tulips'
    ,'bottles'
    ,'bowls'
    ,'cans'
    ,'cups'
    ,'plates'
    ,'apples'
    ,'mushrooms'
    ,'oranges'
    ,'pears'
    ,'sweet peppers'
    ,'clock'
    ,'computer keyboard'
    ,'lamp'
    ,'telephone'
    ,'television'
    ,'bed'
    ,'chair'
    ,'couch'
    ,'table'
    ,'wardrobe'
    ,'bee'
    ,'beetle'
    ,'butterfly'
    ,'caterpillar'
    ,'cockroach'
    ,'bear'
    ,'leopard'
    ,'lion'
    ,'tiger'
    ,'wolf'
    ,'bridge'
    ,'castle'
    ,'house'
    ,'road'
    ,'skyscraper'
    ,'cloud'
    ,'forest'
    ,'mountain'
    ,'plain'
    ,'sea'
    ,'camel'
    ,'cattle'
    ,'chimpanzee'
    ,'elephant'
    ,'kangaroo'
    ,'fox'
    ,'porcupine'
    ,'possum'
    ,'raccoon'
    ,'skunk'
    ,'crab'
    ,'lobster'
    ,'snail'
    ,'spider'
    ,'worm'
    ,'baby'
    ,'boy'
    ,'girl'
    ,'man'
    ,'woman'
    ,'crocodile'
    ,'dinosaur'
    ,'lizard'
    ,'snake'
    ,'turtle'
    ,'hamster'
    ,'mouse'
    ,'rabbit'
    ,'shrew'
    ,'squirrel'
    ,'maple'
    ,'oak'
    ,'palm'
    ,'pine'
    ,'willow'
    ,'bicycle'
    ,'bus'
    ,'motorcycle'
    ,'pickup truck'
    ,'train'
    ,'lawn-mower'
    ,'rocket'
    ,'streetcar'
    ,'tank'
    ,'tractor',
]
#%%

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
#%%
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir="tb_callback_dir", histogram_freq=1,
)

model.fit(
    ds_train,
    epochs=5,
    validation_data=ds_test,
    callbacks=[tensorboard_callback],
    verbose=2,
)

# %%
