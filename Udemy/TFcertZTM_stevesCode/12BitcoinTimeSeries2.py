"""
Created on Wed Aug 24 13:06:02 2021

@author: steve

PLAN:  This is part 2 of 12BitcoinTimeSeries.py

Model 
Num	Model       Type	Horizon Window 	Extra data
0	Naïve model (baseline)	NA	NA	NA
1	tf.Dense model            1	7	NA
2	Same as 1	              1	30	NA
3	Same as 1	              7	30	NA
4	Conv1D	                  1	7	NA
5	LSTM	                  1	7	NA
6	Same as 1 (multivariate ) 1	7	Block reward size
7	N-BEATs Algorithm	      1	7	NA
8	Ensemble (multiple models optimized on different loss functions)	1	7	NA
9	Future prediction model (model to predict future values)	1	7	NA
10	Same as 1 (but with turkey 🦃 data introduced)	1	7	NA

INPUT:  BITCOIN PRICES
OUTPUT: 11 Different models
RUNTIME: 55 sec
"""

#%% starters
import matplotlib.transforms as transforms
import sys, time, os, numpy as np

path = 'C:/Users/steve/Documents/GitHub/misc'
sys.path
sys.path.insert(0,path)
import tensorPrepStarter as tps

starttime, modelstart, filename = tps.setstartingconditions()
import tensorflow as tf
from tensorflow.keras import layers
try:
    filename = os.path.basename(__file__)
except NameError:
    filename = 'working'

#%%  Download Bitcoin historical data from GitHub 
# Note: you'll need to select "Raw" to download the data in the correct format
# to overwrite a file, you HAVE to use a NAME of the file for output.
!c:\users\steve\wget.exe -O BTC_USD_2013to2021.csv \
    https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv 

import pandas as pd
# Parse dates and set date column to index
#df = pd.read_csv("BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv", 
df = pd.read_csv("BTC_USD_2013to2021.csv", 
                 parse_dates=["Date"], 
                 index_col=["Date"]) # parse the date column (tell pandas column 1 is a datetime)
df.tail()
'''The frequency at which a time series value is collected is often referred 
to as seasonality. This is usually mesaured in number of samples per year. 
For example, collecting the price of Bitcoin once per day would result in a 
time series with a seasonality of 365. Time series data collected with 
different seasonality values often exhibit seasonal patterns (e.g. electricity 
demand behing higher in Summer months for air conditioning than Winter months). 
For more on different time series patterns, see
'''

#%% Option #1 Only want closing price for each day 
bitcoin_prices = pd.DataFrame(df["Closing Price (USD)"]).\
    rename(columns={"Closing Price (USD)": "Price"})
bitcoin_prices.head()
import matplotlib.pyplot as plt
# bitcoin_prices.plot(figsize=(10, 7))
# plt.ylabel("BTC Price")
# plt.title("Price of Bitcoin from 1 Oct 2013 to 18 May 2021", fontsize=16)
# plt.legend(fontsize=14);

bitcoin_prices

#%% Shape the Dataframe into two (numpy) arrays, because traintestsplit likes arrays 
# lesson 296...
timesteps = bitcoin_prices.index.to_numpy()  # turn the DF column to an array
prices = bitcoin_prices["Price"].to_numpy()
#prices = bitcoin_prices["Price"].to_numpy(int)

timesteps[:10], prices[:10]

#%% Create train and test splits the right way for time series data
split_size = int(0.8 * len(prices)) # 80% train, 20% test

# Create train data splits (everything before the split)
X_train, y_train = timesteps[:split_size], prices[:split_size]

# Create test data splits (everything after the split)
X_test, y_test = timesteps[split_size:], prices[split_size:]

len(X_train), len(X_test), len(y_train), len(y_test)
print(f'Training points: {len(X_train)} dates, {len(y_train)} prices. \n# of Testing Points: , {len(X_test)}, {len(y_test)}')
#%% DEF plot_time_series (Daniels)
def plot_time_series(timesteps, values, format='.', start=0, end=None, 
                     label=None):
  """
  Plots a timesteps (a series of points in time) against values (a series of values across timesteps).
  
  Parameters
  ---------
  timesteps : array of timesteps
  values : array of values across time
  format : style of plot, default "."
  start : where to start the plot (setting a value will index from start of timesteps & values)
  end : where to end the plot (setting a value will index from end of timesteps & values)
  label : label to show on plot of values
  """
  # Plot the series
  plt.plot(timesteps[start:end], values[start:end], format, label=label)
  plt.xlabel("Time")
  plt.ylabel("BTC Price")
  if label:
    plt.legend(fontsize=14, loc='upper left') # make label bigger
  plt.grid(True)
#%% DEF plot_time_series_model()
def plot_time_series_model(resultsArray, timesteps_in, offset, plttitle, model_label_desc,
                           predictions):
    ''' 
    '''
    rslts0 = str(resultsArray).replace(',','\n')
    offset = offset # offset the values by 300 timesteps
    plt.title(plttitle, fontsize=16)
    plot_time_series(timesteps=X_test, values=y_test, start=offset, 
                     label="Dots are X_test ie: Bitcoin prices")
#    plot_time_series(timesteps=X_test[1:], values=predictions, 
    plot_time_series(timesteps=timesteps_in, values=predictions, 
                     format="-", start=offset, label=model_label_desc);

#%% DEF get_labeled_windows(x, horizon=1)
HORIZON = 7 # predict 1 step at a time
WINDOW_SIZE = 7 # use a week worth of timesteps to predict the horizon
# Create function to label windowed data
def get_labelled_windows(x, horizon=1):
  """
  Creates labels for windowed dataset.

  E.g. if horizon=1 (default)
  Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
  """
  return x[:, :-horizon], x[:, -horizon:]

# Test out the window labelling function
test_window, test_label = \
    get_labelled_windows(tf.expand_dims(tf.range(8)+1, axis=0), horizon=HORIZON)
# Below, we 'squeeze' to remove the empty dimension...
# that has the effect of 'pulling out' an embedded data inside a lower bracket
print(f"Window: {tf.squeeze(test_window).numpy()} -> Label: {tf.squeeze(test_label).numpy()}")
#%% DEF make_windows  Create function to view NumPy arrays as windows 
def make_windows(x, window_size=7, horizon=1):
  """
  Turns a 1D array into a 2D array of sequential windows of window_size.
  """
  # 1. Create a window of specific window_size (add the horizon on the end for later labelling)
  window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
  # print(f"Window step:\n {window_step}")

  # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
  window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T # create 2D array of windows of size window_size
  # print(f"Window indexes:\n {window_indexes[:3], window_indexes[-3:], window_indexes.shape}")

  # 3. Index on the target array (time series) with 2D array of multiple window steps
  windowed_array = x[window_indexes]

  # 4. Get the labelled windows
  windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

  return windows, labels

full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
len(full_windows), len(full_labels)
#%% DEF make_train_test_splits(windows, labels, test_split=0.2) 
def make_train_test_splits(windows, labels, test_split=0.2):
  """
  Splits matching pairs of windows and labels into train and test splits.
  """
  split_size = int(len(windows) * (1-test_split)) #default to 80% train/20% test
  train_windows = windows[:split_size]  #train_windows.shape (2224, 7)
  train_labels = labels[:split_size]    #train_labels.shape  (2224, 1)
  test_windows = windows[split_size:]  # test_windows.shape  (556, 7)
  test_labels = labels[split_size:]     #test_labels.shape   (556, 1)
  return train_windows, test_windows, train_labels, test_labels  
#%% DEF create_model_checkpoint(model_name, save_path="model_experiments")
import os
# Create a function to implement a ModelCheckpoint callback with a specific filename 
# If we don't specify otherwise, 'validation_loss' is the thing to measure for 
#   'success'... and that will determine which model is saved.
def create_model_checkpoint(model_name, save_path="model_experiments"):
  return tf.keras.callbacks.\
      ModelCheckpoint(filepath=os.path.join(save_path, model_name), # create filepath to save model
                      verbose=0, # only output a limited amount of text
                      save_best_only=True) # save only the best model to file
#%% DEF make_preds(model, input_data)
def make_preds(model, input_data):
  """
  Uses model to make predictions on input_data.

  Parameters
  ----------
  model: trained model 
  input_data: windowed input data (same kind of data model was trained on)

  Returns model predictions on input_data.
  """
  forecast = model.predict(input_data)
  return tf.squeeze(forecast) # return 1D array of predictions


#%% DEF mean_absolute_scaled_error Calculate 5 metrics
# MASE implemented courtesy of sktime - https://github.com/alan-turing-institute/sktime/blob/ee7a06843a44f4aaec7582d847e36073a9ab0566/sktime/performance_metrics/forecasting/_functions.py#L16
def mean_absolute_scaled_error(y_true, y_pred):
  """
  Implement MASE (assuming no seasonality of data).
  """
  mae = tf.reduce_mean(tf.abs(y_true - y_pred))

  # Find MAE of naive forecast (no seasonality)
  mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1])) # our seasonality is 1 day (hence the shifting of 1 day)

  return mae / mae_naive_no_season
#%% DEF evaluate_preds
def evaluate_preds(y_true, y_pred):
  # Make sure float32 (for metric calculations)
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)

  # Calculate various metrics  (each made as  tensorflow.python.framework.ops.EagerTensor)
  mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
  mse = tf.keras.metrics.mean_squared_error(y_true, y_pred) # puts and emphasis on outliers (all errors get squared)
  rmse = tf.sqrt(mse)
  mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
  mase = mean_absolute_scaled_error(y_true, y_pred)
    # Account for different sized metrics (for longer horizons, reduce to single number)
  if mae.ndim > 0: # if mae isn't already a scalar, reduce it to one by aggregating tensors to mean
    mae = tf.reduce_mean(mae)
    mse = tf.reduce_mean(mse)
    rmse = tf.reduce_mean(rmse)
    mape = tf.reduce_mean(mape)
    mase = tf.reduce_mean(mase)

  return {"mae": np.round(mae.numpy(),1),  # .numpy() converts from EagerTensor to float64
          "mse": np.round(mse.numpy(),1),
          "rmse": np.round(rmse.numpy(),1),
          "mape": np.round(mape.numpy(),1),
          "mase": mase.numpy()}
#%% MODEL 3 CREATE, compile, fit:  DENSE with window size of 30
HORIZON = 7 # predict one step at a time
WINDOW_SIZE = 30 # use 30 timesteps in the past

# Make windowed data with appropriate horizon and window sizes
full_windows30, full_labels30 = make_windows(prices, window_size=WINDOW_SIZE, 
                                             horizon=HORIZON)
len(full_windows30), len(full_labels30)

# Make train and testing windows
train_windows30, test_windows30, train_labels30, test_labels30 = \
    make_train_test_splits(windows=full_windows30, labels=full_labels30)
len(train_windows30), len(test_windows30), len(train_labels30), len(test_labels30)

tf.random.set_seed(42)
# Create model (same model as model 1 but data input will be different)
model_3 = tf.keras.Sequential([
  layers.Dense(128, activation="relu"),
  layers.Dense(HORIZON) # need to predict horizon number of steps into the future
], name="model_3_dense")
model_3.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam())
model_3.fit(train_windows30,
            train_labels30,
            epochs=100,
            batch_size=128,
            verbose=0,
            validation_data=(test_windows30, test_labels30),
            callbacks=[create_model_checkpoint(model_name=model_3.name)])
model_3.summary()
#%% # Evaluate model 3 preds
model_3.evaluate(test_windows30, test_labels30)

# Load in best performing model
model_3 = tf.keras.models.load_model("model_experiments/model_3_dense/")
model_3.evaluate(test_windows30, test_labels30)

#%% # Get forecast predictions
model_3_preds = make_preds(model_3,
                           input_data=test_windows30)
# Evaluate results for model 3 predictions
model_3_results = evaluate_preds(y_true=tf.squeeze(test_labels30), # remove 1 dimension of test labels
                                 y_pred=model_3_preds)
model_3_results
#%% MODEL 3 chart... 
rslts3 = str(model_3_results).replace(',','\n')
fig, ax = plt.subplots(figsize=( 10, 10))
trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
offset = 300
plot_time_series(timesteps=X_test[-len(test_windows30):], values=test_labels30[:, 0], 
                 start=offset, label="test_data")
plot_time_series(timesteps=X_test[-len(test_windows30):], 
                 values=tf.reduce_mean(model_3_preds, axis=1), 
                 start=offset, format="-", 
                 label="MDL3: tf.dense.2hidlayers Horiz 7 window 30") 
#ax.text(0.5, 0.5, "middle of graph", transform=trans)
plt.text(.3, .1, horizontalalignment='right', transform=trans, va='bottom',
         s=rslts3, wrap=True,
         fontsize=14, bbox=dict(facecolor='pink', alpha=0.5))
plt.show()
#%% Model 4 Conv1d... Prepare data
HORIZON = 1
WINDOW_SIZE = 7
full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
len(full_windows), len(full_labels)
train_windows, test_windows, train_labels, test_labels = \
    make_train_test_splits(windows=full_windows, labels=full_labels)
len(train_windows), len(test_windows), len(train_labels), len(test_labels)
print(f"Training Windows: {tf.squeeze(train_windows).numpy().shape}  -> Label: {tf.squeeze(train_labels).numpy().shape}")
print(f"Testing Windows: {tf.squeeze(test_windows).numpy().shape}  -> Label: {tf.squeeze(test_labels).numpy().shape}")
#%% Model 4 prep data more...
# Before we pass our data to the Conv1D layer, we have to reshape it in order 
#  to make sure it works
train_windows[0] # let look at our first row...
x = tf.constant(train_windows[0])
expand_dims_layer = layers.Lambda(lambda x: tf.expand_dims(x, axis=1)) # add an extra dimension for timesteps
print(f"Original shape: {x.shape}") # (WINDOW_SIZE)
print(f"Expanded shape: {expand_dims_layer(x).shape}") # (WINDOW_SIZE, input_dim) 
print(f"Original values with expanded shape:\n {expand_dims_layer(x)}")
x.numpy()
#%% # Create model #4 (Conv1d)
mdl4start = time.perf_counter()
model_4 = tf.keras.Sequential([
  # Create Lambda layer to reshape inputs, without this layer, the model will error
  # resize the inputs to adjust for window size / Conv1D 3D input requirements
  layers.Lambda(lambda x: tf.expand_dims(x, axis=1)), 
  layers.Conv1D(filters=128, kernel_size=5, padding="causal", activation="relu"),
  layers.Dense(HORIZON)
], name="model_4_Conv1d")
model_4.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam())
model_4.fit(train_windows,
            train_labels,
            epochs=100,
            batch_size=128,
            verbose=0,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_name=model_4.name)])
mdl4duration = time.perf_counter() - mdl4start
model_4.summary()
#%% # Evaluate model 4 preds
model_4.evaluate(test_windows, test_labels)

# Load in best performing model
model_4 = tf.keras.models.load_model("model_experiments/model_4_Conv1d/")
model_4.evaluate(test_windows, test_labels)

#%% # Get forecast predictions
model_4_preds = make_preds(model_4,
                           input_data=test_windows)
# Evaluate results for model 4 predictions
model_4_results = evaluate_preds(y_true=tf.squeeze(test_labels), # remove 1 dimension of test labels
                                 y_pred=model_4_preds)
model_4_results
#%% MODEL 4 chart... 
rslts4 = str(model_4_results).replace(',','\n')
fig, ax = plt.subplots(figsize=( 10, 10))
trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
offset = 300
plot_time_series(timesteps=X_test[-len(test_windows):], 
                 values=test_labels[:, 0], 
                 start=offset, label="test_data")
plot_time_series(timesteps=X_test[-len(test_windows):], 
                 values=model_4_preds, 
                 start=offset, format="-", 
                 label=f'MDL4: tf.Conv1d.2hidlayers Horizon: {HORIZON} window {WINDOW_SIZE}') 
#ax.text(0.5, 0.5, "middle of graph", transform=trans)
plt.text(.3, .1, horizontalalignment='right', transform=trans, va='bottom',
         s=rslts4, wrap=True,
         fontsize=14, bbox=dict(facecolor='pink', alpha=0.5))

#%% Model 5 RNN (LSTM) v2
mdl5start = time.perf_counter()
inputs = layers.Input(shape = (WINDOW_SIZE))
x = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)
#x = layers.LSTM(128, return_sequences=TRUE)(x)
x = layers.LSTM(128, activation='relu')(x)
#x = layers.Dense(32, activation='relu')(x)
output = layers.Dense(HORIZON)(x)
model_5 = tf.keras.Model(inputs=inputs, outputs=output, name="model_5_LSTM")
model_5.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam())
model_5.fit(train_windows,
            train_labels,
            epochs=100,
            batch_size=128,
            verbose=1,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_name=model_5.name)])
mdl5duration = time.perf_counter() - mdl5start
model_5.summary()

#%% # Evaluate model 5 forecast
model_5.evaluate(test_windows, test_labels)

# Load in best performing model
model_5 = tf.keras.models.load_model("model_experiments/model_5_LSTM")
model_5.evaluate(test_windows, test_labels)
#%% # Get forecast predictions
model_5_preds = make_preds(model_5,
                           input_data=test_windows)
# Evaluate results for model 5 predictions
model_5_results = evaluate_preds(y_true=tf.squeeze(test_labels), # remove 1 dimension of test labels
                                 y_pred=model_5_preds)
model_5_results
#%% Model 5 chart
rslts5 = str(model_5_results).replace(',','\n')
fig, ax = plt.subplots(figsize=( 10, 10))
trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
offset = 300
plot_time_series(timesteps=X_test[-len(test_windows):], 
                 values=test_labels[:, 0], 
                 start=offset, label="test_data")
plot_time_series(timesteps=X_test[-len(test_windows):], 
                 values=model_5_preds, 
                 start=offset, format="-", 
                 label=f'MDL5: tf.LSTM.2hidlayers Horizon: {HORIZON} window {WINDOW_SIZE}') 
#ax.text(0.5, 0.5, "middle of graph", transform=trans)
plt.text(.3, .1, horizontalalignment='right', transform=trans, va='bottom',
         s=rslts5, wrap=True,
         fontsize=14, bbox=dict(facecolor='pink', alpha=0.5))
#%% ENDING
duration =  time.perf_counter() - starttime
if duration < 60:
    print('\nit  took: {:.2f} seconds\n'.format(duration))
    durationtm = str(round(duration,2)) + ' sec'
elif duration >= 60: 
    print('\nIt  took: {:.2f} minutes\n'.format(duration/60))
    duration = round(duration/60,2)
    durationtm = str(duration) + ' min'
#%% Put the charts together for comparison...
#plt.figure(figsize=(12,6))
#plt.figure(figsize = (20,10), tight_layout=None)
datadesc = f'Training points: {len(X_train)} dates, {len(y_train)} prices.'\
    f' \n# of Testing Points: , {len(X_test)}, {len(y_test)}' \
        f'\nBut forecast is short 1 ie: {len(model_5_preds)}'
mdl1note1 = '''With these outputs, our model isn't forecasting yet. It's only making 
predictions on the test dataset. Forecasting would involve a model 
making predictions into the future, however, the test dataset is 
only a pseudofuture.'''
notes =  'first 3 Bitcoin Price windows:'
notes0 = f"\nWindow: {full_windows[0]} -> Label: {full_labels[0]}"
notes1 = f"\nWindow: {full_windows[1]} -> Label: {full_labels[1]}"
notes2 = f"\nWindow: {full_windows[2]} -> Label: {full_labels[2]}"
# https://www.machinelearningplus.com/plots/matplotlib-pyplot/
fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots\
   (2, 2, sharey=True, figsize = (20,20), sharex=True)
plt.subplots_adjust(wspace =.05, hspace = .08)  
plt.suptitle(f'UDEMY TFCertZ2M {filename}   {modelstart}  Duration: {durationtm} '\
             '\nBitcoin price Actual & Forecast by '\
             'various models\nNotice shared AxisLabels, reduced margins'\
            ' &  Notice that the best was the NAIVE model!!!'\
                '\nSec 12 Lessons 313 thru 319'
             , fontsize=20, weight='bold')

#ax = fig.add_subplot(121)
plt.subplot(2,2,1)
trans1 = transforms.blended_transform_factory(ax1.transAxes, ax1.transAxes)
ax1.text(.3, .3, horizontalalignment='right', transform=trans1, 
         va='bottom', s=rslts3, wrap=True,
          fontsize=14, bbox=dict(facecolor='pink', alpha=0.5))
ax1.text(.03, .5, horizontalalignment='left', transform=trans1, 
         va='bottom', s= notes + notes0 + notes1 + notes2,
         fontsize=16, bbox=dict(facecolor='green', alpha=0.2))
ax1.text(.03, .73, horizontalalignment='left', transform=trans1, 
         va='bottom', s=datadesc, wrap=True,
          fontsize=14, bbox=dict(facecolor='blue', alpha=0.3))
ax1.text(.23, .03, horizontalalignment='left', transform=trans1, 
         va='bottom', s=tps.mdl2txt(model_3), wrap=True,
          fontsize=14, bbox=dict(facecolor='blue', alpha=0.3))
plot_time_series_model(model_3_results,  X_test[-len(test_windows30):], 300, 
                       f'{model_3.name} - (Orange Line)',
                       f'Mdl3: WindowSize 30 Horizon: 7  ',
                       tf.reduce_mean(model_3_preds, axis=1))
plt.subplot(2,2,2)
trans2 = transforms.blended_transform_factory(ax2.transAxes, ax2.transAxes)
ax2.text(.3, .3, horizontalalignment='right', transform=trans2, 
         va='bottom', s=rslts4, wrap=True,
          fontsize=14, bbox=dict(facecolor='pink', alpha=0.5))
# ax1.text(.03, .5, horizontalalignment='left', transform=trans1, 
#          va='bottom', s= notes + notes0 + notes1 + notes2,
#           fontsize=16, bbox=dict(facecolor='green', alpha=0.2))
ax2.text(.03, .73, horizontalalignment='left', transform=trans2, 
         va='bottom', s=datadesc, wrap=True,
          fontsize=14, bbox=dict(facecolor='blue', alpha=0.3))
ax2.text(.23, .03, horizontalalignment='left', transform=trans2, 
         va='bottom', s=tps.mdl2txt(model_4), wrap=True,
          fontsize=14, bbox=dict(facecolor='blue', alpha=0.3))
plot_time_series_model(model_4_results,  X_test[-len(test_windows):], 300, 
                       f'{model_4.name}- (Orange Line)  Time: {round(mdl4duration,2)} sec', 
                       f'MDL4: Conv1d  WindowSize: {WINDOW_SIZE}; Horizon: {HORIZON}',
                       model_4_preds)
plt.subplot(2,2,3)
trans3 = transforms.blended_transform_factory(ax3.transAxes, ax3.transAxes)
ax3.text(.3, .3, horizontalalignment='right', transform=trans3, 
         va='bottom', s=rslts5, wrap=True,
          fontsize=14, bbox=dict(facecolor='pink', alpha=0.5))
ax3.text(.03, .5, horizontalalignment='left', transform=trans3, 
         va='bottom', s=mdl1note1, wrap=True,
          fontsize=14, bbox=dict(facecolor='green', alpha=0.2))
ax3.text(.3, .03, horizontalalignment='left', transform=trans3, 
         va='bottom', s=tps.mdl2txt(model_5), wrap=True,
          fontsize=14, bbox=dict(facecolor='blue', alpha=0.3))
plot_time_series_model(model_5_results,  X_test[-len(test_windows):], 300, 
                       f'{model_5.name}:(Orange Line)   Time: {round(mdl5duration,2)} sec', 
                       f'MDL5: tf.LSTM.2hidlayers WindowSize: {WINDOW_SIZE}; Horizon: {HORIZON}', 
                       model_5_preds)

plt.show()
#%%
def mdl2txt(model_in):
    stringlist2 = []
    model_in.summary(print_fn=lambda x: stringlist2.append(x))
    model_insum = "\n".join(stringlist2)
    model_insum = model_insum.replace('_________________________________________________________________\n', '')
    model_insum = model_insum.replace('=','')
    model_insum = model_insum.replace('_','')
    # model_insum = model_insum.replace('\n\n','\n')
    # model_insum = model_insum.replace('\\n\'','\'')
    return model_insum

mdl2txt(model_4)
