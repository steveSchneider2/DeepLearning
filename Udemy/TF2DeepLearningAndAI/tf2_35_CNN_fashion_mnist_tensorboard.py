# -*- coding: utf-8 -*-
"""TF2.0 Fashion MNIST.ipynb
13 June: Adding Tensorboard callbacks.
21 July: Added the 'Projector' callback! Yeay!

For Callbacks and making tensorboard work... (that's right neputune shows how)
 https://neptune.ai/blog/tensorboard-tutorial
 
via TERMINAL command:  tensorboard --logdir d:\data\logs\mnist20210614_143149
Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XeVQQdyGNptGWBLclF3yVijNir5-upkF
    
12 Jan: Fully functional in Environment: P37TF22cu7  (actually TF21)

Convolution 
    --> Image modification...either addition or multiplication (or both)
    --> ie feature transformation... 
        like taking just the lines from an image (edge detection filter)
        like bluring a picture (gaussian filter)

alternative description of convolution:
    "A sliding pattern finder (looking for particular pattern)"
    

UDEMY TENSORFLOW 2.0 LECTURE 35  MNIST IMAGE CLASSIFICATION FASHION 60,000 RECORDS
RUNNING ON GPU: TAKES 107.8 SECONDS 

  Notes from the lecture #35:
It's Conv23 because there are two spatial dimensions; color is not a spatial dimension'
A time-varing signal (sound) would use Conv1D... only dimension is sound
A video (height, Width, Time) would use Conv3d
Medical imaging data (like cancer images) would have height, width, depth (use Conv3D)
Conv3D would use Voxel(s) (volumn element vs picture element)

# output feature maps (32) 
Filter dimensions (3, 3) ... the spatial dimensions
Stride controls the speed of the filter
Activiation 
Padding arguement; default is 'valid'... which comes w/o an entry.

Convolution is pattern finding...so you might not usually do dropouts in convolution...
    because you don't want to remove pixels before looking for the pattern (stroke)
    if you remove pixels, is the 'stroke' still visible
Tensor24 env:
2021-06-13 14:14:46.040247: E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1415] function cupti_interface_->Subscribe( &subscriber_, (CUpti_CallbackFunc)ApiCallback, this)failed with error CUPTI could not be loaded or symbol could not be found.
2021-06-13 14:14:46.040300: I tensorflow/core/profiler/lib/profiler_session.cc:172] Profiler session tear down.
2021-06-13 14:14:46.040326: E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1496] function cupti_interface_->Finalize()failed with error CUPTI could not be loaded or symbol could not be found.


# Commented out IPython magic to ensure Python compatibility.
# Install TensorFlow
# !pip install -q tensorflow-gpu==2.0.0-beta1

"""
#%% Imports...
import numpy 
import GPUtil
import numpy as np, matplotlib.pyplot as plt, os,pandas as pd
# 1 July 2021... next two statements...BEFORE any tensorflow did the trick.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time, sys, tensorflow as tf
from datetime import datetime

import tensorboard ,sklearn.metrics, itertools, io
from tensorboard.plugins import projector, hparams as hp
# from tensorboard.plugins import profiler_session
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import pydot, graphviz  #for graph of 
gpus = GPUtil.getGPUs()

#%% Get GPU status
from tensorflow.python.client import device_lib 
# print(device_lib.list_local_devices())  # this puts out a lot of lines (Gibberish?)
print('Conda Envronment:  ', os.environ['CONDA_DEFAULT_ENV'])
print(f'Gpu  Support:       {tf.test.is_built_with_gpu_support()}')
print(f'Cuda Support:       {tf.test.is_built_with_cuda()}')
print(f'Tensor Flow:        {tf.version.VERSION}')
pver = str(format(sys.version_info.major) +'.'+ format(sys.version_info.minor)+'.'+ format(sys.version_info.micro))
print('Python version:      {}.'.format(pver)) 
print('The numpy version:   {}.'.format(np.__version__))
print('The panda version:   {}.'.format(pd.__version__))
print('Tensorboard version  {}.'.format(tensorboard.__version__))
# additional imports

condaenv = os.environ['CONDA_DEFAULT_ENV']

modelstart = time.strftime('%c')
#%%  # Define the Keras TensorBoard callback.
label_dict = {  # NOT USED...but it adds clarity to see it.
            0:	'T-shirt/top',
            1:	'Trouser',
            2:	'Pullover',
            3:	'Dress',
            4:	'Coat',
            5:	'Sandal',
            6:	'Shirt',
            7:	'Sneaker',
            8:	'Bag',
            9:	'Ankle boot'
           }
logdir="d:/data/logs/hadelin/mnist/" + datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_callback = keras.callbacks.\
    TensorBoard(log_dir=logdir,histogram_freq=1,  write_graph=True,
                write_images=True, update_freq='epoch',
                profile_batch=6, embeddings_freq=1,
#                embeddings_metadata={layer_embed.name:'./logs/text_classify/word.tsv'})
#                embeddings_metadata='D:/Data/logs/hadelin/mnist/classes.tsv')
                embeddings_metadata='classes.tsv')
#              profile_batch='500,520')  #this seemed to fix the errors noted in the opening dialog above. :) 
file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')

#%% Load in the data
# Notice below, the dataset come INSIDE the tensorflow module...
# the 'load_data()' function returns two tuples.
# x_train is a numpy array (60000,  28, 28, 1); y_train(60000,)
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
(x_img, y_img), (x_imgt, y_imgt) = fashion_mnist.load_data()

# Names of the integer classes, i.e., 0 -> T-short/top, 1 -> Trouser, etc.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#%%  # Creates a file writer for the log directory.
imagedir = logdir+'/imgs'
file_writer = tf.summary.create_file_writer(imagedir)

with file_writer.as_default():
  # Don't forget to reshape.
  images = np.reshape(x_train[0:25], (-1, 28, 28, 1))
  tf.summary.image("25 training data examples", images, max_outputs=25, step=0)

test_summary_writer = tf.summary.create_file_writer(logdir)
with test_summary_writer.as_default():
    tf.summary.scalar('loss', 0.345, step=1)
    tf.summary.scalar('loss', 0.234, step=2)
    tf.summary.scalar('loss', 0.123, step=3)  
#%% def plot_confusion_matrix
def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()
  return figure
#%% def plot_to_image(figure):
def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image
#%%  def log_confusion_matrix(epoch, logs):
def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
  test_pred_raw = mnistMDL.predict(x_test) #ie... (test_images)
  test_pred = np.argmax(test_pred_raw, axis=1)

  # Calculate the confusion matrix.
  cm = sklearn.metrics.confusion_matrix(y_test, test_pred)
  # Log the confusion matrix as an image summary.
  figure = plot_confusion_matrix(cm, class_names=class_names)
  cm_image = plot_to_image(figure)

  # Log the confusion matrix as an image summary.
  with file_writer_cm.as_default():
    tf.summary.image("Confusion Matrix", cm_image, step=epoch)

# Define the per-epoch callback.
cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

#%% Normalization / Data Prep
x_train, x_test = x_train / 255.0, x_test / 255.0
print("x_train.shape:", x_train.shape)  # (60000, 28, 28)
type(x_train)
# the data is only 2D!
# convolution expects height x width x color
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(x_train.shape)

# number of classes...cast y_train to a 'set', which only has unique values
K = len(set(y_train))
print("number of classes:", K)  # ie 10 classes; we knew this already but good to show the general case
set(y_train)  # ie:  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

# ####################################################
#%% BUILD THE MODEL and compile/fit using the functional API
i = Input(shape=x_train[0].shape)
# convolution layers (3 of them): increasing the #of feature maps at each convolution layer
# This (doubling feature maps) is a 'pattern'... (he says)
#  Strides of 2 reduces the image dimension by half after each convolution --> he says this as if it is purposeful, by design
# Convolution layers do feature extraction; 
x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
# Convolution done...now just a 'simple' ANN... 'classification'
# And, that means...flatten() first
x = Flatten()(x)
x = Dropout(0.2)(x)  # for regularization
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

mnistMDL = Model(i, x)

# Note: make sure you are using the GPU for this!
# mnistMDL.compile(optimizer='adam',
mnistMDL.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#%% Train the model
starttime = time.perf_counter()
epochs = 12
r = mnistMDL.fit(x_train, y_train, validation_data=(x_test, y_test), 
              epochs=epochs, callbacks=[tensorboard_callback, cm_callback])
endtime = time.perf_counter()
duration = round(endtime - starttime,2)
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')
#%% Graph the model
# import pydot
# import graphviz
from tensorflow import keras
# from tensorflow.keras import layers
keras.utils.plot_model(mnistMDL, show_shapes=True)
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')

#%% Show the model in text
s = io.StringIO()
mnistMDL.summary(print_fn=lambda x: s.write(x + '\n'))
model_summary = s.getvalue()
model_summary = model_summary.replace('=','')
model_summary = model_summary.replace('_','')
model_summary = model_summary.replace('\n\n','\n')

pver = str(format(sys.version_info.major) + '.' +
           format(sys.version_info.minor) + '.' +
           format(sys.version_info.micro))
mnistMDL.summary()
#%% Plot loss per iteration multi-color chart!
s2= """i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)"""

plt.style.available

plt.style.use('classic')
plt.style.use('ggplot')
#plt.style.use('seaborn')  # 22 July...this looks best; grey face; white background
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

# Plot accuracy per iteration
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title(f'Udemy TensorFlow 2.0 Lecture 35 {modelstart}\n ' +
          f'Image Classification: Fashion MNIST {x_train.shape[0]} records ' +
          f'{x_train.shape[1]} size')
plt.xlabel(f'tf2_35_CNN_fashion_mnist.py  Convolutional NN  Duration: {duration}')
plt.ylim(0,1)
# plt.text(.5, .7,# transform=trans1,
#          s=s2,
#          wrap=True, ha='left', va='bottom',
#          fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))
plt.text(.5, .2,# transform=trans1,
         s=model_summary,
         wrap=True, ha='left', va='bottom', fontname='Consolas',
         fontsize=8, bbox=dict(facecolor='pink', alpha=0.5))
plt.text(.5, .02,# transform=trans1,
         s=f'Conda Envr:  {condaenv}\n' +
         f'Gpu  Support:       {tf.test.is_built_with_gpu_support()}\n' +
         f'Python: {pver} tensorflow Ver: {tf.version.VERSION}' +
         '\nConvolutional neural network' + 
         f'\n{epochs} epochs, Duration: {duration:3.2f} seconds',# +
#         f'\n{gpus[0].name} Cuda 11.1.relgpu',
         wrap=True, ha='left', va='bottom',
         fontsize=9, bbox=dict(facecolor='aqua', alpha=0.5))
plt.text(.2, .7, s='''I returned to this on 21 July to implement more complete
         tensorboarding.  TB's Scalars, images, graphs, distributions
         histograms and PROJECTOR (3 dimensionally dynamic!!!)''',
         wrap=True, ha='left', va='bottom',
         fontsize=9, bbox=dict(facecolor='green', alpha=0.5))
plt.show()

#%% Plot confusion matrix
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix2(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  #  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=35)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

p_test = mnistMDL.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, p_test)

# plot_confusion_matrix(cm, list(range(10)))
# Label mapping
labels = '''0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot'''.split("\n")

plot_confusion_matrix(cm, labels)
plot_confusion_matrix2(cm, labels)

#%% Show some misclassified examples
misclassified_idx = np.where(p_test != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i].reshape(28,28), cmap='gray')
plt.title("True label: %s Predicted: %s" % (labels[y_test[i]], labels[p_test[i]]));

#%%
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')

#%%
'''
#%% Still to make this work...#%%
# https://github.com/sachinruk/tensorboard_vis
# another way: https://github.com/anujshah1003/Tensorboard-own-image-data-image-features-embedding-visualization
label_dict = {
            0:	'T-shirt/top',
            1:	'Trouser',
            2:	'Pullover',
            3:	'Dress',
            4:	'Coat',
            5:	'Sandal',
            6:	'Shirt',
            7:	'Sneaker',
            8:	'Bag',
            9:	'Ankle boot'
           }
from collections import Counter
destination = 'd:/data/udemy/hadelin/train-images-idx3-ubyte'
destinatio2 = 'd:/data/udemy/hadelin/train-labels-idx1-ubyte'
with open(destination, 'rb') as imgpath:
    images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape((-1,28,28))
    
with open(destinatio2, 'rb') as imgpath:
    im_labels = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=8)
len(im_labels)    # 60,000 labels
Counter(im_labels).most_common()

destinatio3 = 'd:/data/udemy/hadelin/'
with open(os.path.join(destinatio3, 'metadata.tsv'), 'w') as f:
    f.write('Class\tName\n')
    for num, name in zip(im_labels, [label_dict[l] for l in im_labels]):
        f.write('{}\t{}\n'.format(num,name))


def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding
    Args:
      data: NxHxW[x3] tensor containing the images.
    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data

TO_EMBED_COUNT = 8000
batch_xs, batch_ys = x_train[:TO_EMBED_COUNT, :], y_train[:TO_EMBED_COUNT]
with open("D:\Data\logs\hadelin\\mnist\\20210721_010820\\vectors.tsv", 'w') as f:
    for image in batch_xs:
        image = image.flatten()
        f.write("\t".join([str(a) for a in image]))
        f.write("\n")
with open('D:\Data\logs\hadelin\mnist\\20210721_010820\labels.tsv', 'w') as f:
    for label in batch_ys: 
        f.write(str(label) + "\n")
        
        
# import scipy
sprite = images_to_sprite(images)
len(sprite)
#scipy.misc.imsave(os.path.join(destinatio3, 'sprite.png'), sprite)
plt.imsave(os.path.join(destinatio3, 'sprite.png'), sprite)
sprite.shape

features = tf.Variable(images.reshape((len(images), -1)), name='features')

# Save the weights we want to analyze as a variable. Note that the first
# value represents any unknown word, which is not in the metadata, here
# we will remove this value.
weights = tf.Variable(mnistMDL.layers[3].get_weights()[0][1:])
# Create a checkpoint from embedding, the filename and key are the
# name of the tensor.
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint = tf.train.Checkpoint(embedding=x_img[0][1:])
checkpoint.save(os.path.join(logdir, "embedding.ckpt"))

# Set up config.
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(logdir, config)



# Iterate thru all the layers of the model
for layer in mnistMDL.layers:
    if 'conv' in layer.name:
        weights, bias= layer.get_weights()
        print(layer.name, filters.shape)
        
        #normalize filter values between  0 and 1 for visualization
        f_min, f_max = weights.min(), weights.max()
        filters = (weights - f_min) / (f_max - f_min)  
        print(filters.shape[3])
        filter_cnt=1
        
        #plotting all the filters
        for i in range(filters.shape[3]):
            #get the filters
            filt=filters[:,:,:, i]
            #plotting each of the channel, color image RGB channels
            for j in range(filters.shape[0]):
                ax= plt.subplot(filters.shape[3], filters.shape[0], filter_cnt  )
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(filt[:,:, j])
                filter_cnt+=1
        plt.show()
'''        