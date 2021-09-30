# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 17:06:02 2021

@author: steve

PLAN:

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
import sys, time, os

path = 'C:/Users/steve/Documents/GitHub/misc'
sys.path
sys.path.insert(0,path)
import tensorPrepStarter as tps

starttime, modelstart, filename = tps.setstartingconditions()
import tensorflow as tf
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
''' Python’s Tilde ~n operator is the bitwise negation operator: it takes the number n as binary number and “flips” all bits 0 to 1 and 1 to 0 to obtain the complement binary number. 
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

#%% Option #2
# Importing and formatting historical Bitcoin data with Python
import csv
from datetime import datetime

timesteps = []
btc_price = []
with open("BTC_USD_2013to2021.csv", "r") as f:
  csv_reader = csv.reader(f, delimiter=",") # read in the target CSV
  next(csv_reader) # skip first line (this gets rid of the column titles)
  for line in csv_reader:
    timesteps.append(datetime.strptime(line[1], "%Y-%m-%d")) # get the dates as dates (not strings), strptime = string parse time
    btc_price.append(float(line[2])) # get the closing price as float

# View first 10 of each
timesteps[:10], btc_price[:10]

# Plot from CSV
import matplotlib.pyplot as plt
import numpy as np
# plt.figure(figsize=(10, 7))
# plt.plot(timesteps, btc_price)
# plt.title("Price of Bitcoin from 1 Oct 2013 to 18 May 2021", fontsize=16)
# plt.xlabel("Date")
# #plt.legend(fontsize=14)
# plt.ylabel("BTC Price");

#%% Shape the Dataframe into two (numpy) arrays, because traintestsplit likes arrays 
# lesson 296...
timesteps = bitcoin_prices.index.to_numpy()  # turn the DF column to an array
prices = bitcoin_prices["Price"].to_numpy(int)

timesteps[:10], prices[:10]

#%% Create train and test splits the right way for time series data
split_size = int(0.8 * len(prices)) # 80% train, 20% test

# Create train data splits (everything before the split)
X_train, y_train = timesteps[:split_size], prices[:split_size]

# Create test data splits (everything after the split)
X_test, y_test = timesteps[split_size:], prices[split_size:]

len(X_train), len(X_test), len(y_train), len(y_test)
print(f'Training points: {len(X_train)} dates, {len(y_train)} prices. \n# of Testing Points: , {len(X_test)}, {len(y_test)}')
#%% Plot correctly made splits
plt.figure(figsize=(10, 7))
plt.title('Plot the Bitcoin values split')
plt.scatter(X_train, y_train, s=5, color='blue', label="Train data")
plt.scatter(X_test, y_test, s=5, color='red', label="Test data")
plt.xlabel("Date")
plt.ylabel("BTC Price")
plt.legend(fontsize=14)
plt.show();

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
#%% Try out our plotting function
# plt.figure(figsize=(10, 7))
# plot_time_series(timesteps=X_train, values=y_train, label="Train data")
# plot_time_series(timesteps=X_test, values=y_test, label="Test data")
#%% MODEL 0 NAIVE PREDICTIONS... Create a naïve forecast
# Naïve forecast equals every value excluding the last value
naive_forecast = np.round(y_test[:-1] ,2)
naive_forecast[:10], naive_forecast[-10:] # View frist 10 and last 10 

#%% # Plot naive forecast
plt.figure(figsize=(10, 7))
plot_time_series(timesteps=X_train, values=y_train, label="Train data")
plot_time_series(timesteps=X_test, values=y_test, label="Test data")
plot_time_series(timesteps=X_test[1:], values=naive_forecast, format="-", 
                 label="Naive forecast");

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
  
  return {"mae": round(mae.numpy(),1),  # .numpy() converts from EagerTensor to float64
          "mse": round(mse.numpy(),1),
          "rmse": round(rmse.numpy(),1),
          "mape": round(mape.numpy(),1),
          "mase": mase.numpy()}
#%% MODEL0 NAIVE_results: 5 metrics: mae, mse, rmse, mape
naive_results = evaluate_preds(y_true=y_test[1:],
                               y_pred=naive_forecast)
naive_results
#%% MODEL0 plot NAIVE
rslts0 = str(naive_results).replace(',','\n')
# rslts0 = str(rslts0.replace(':','\/t'))
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
#fig, ax = plt.subplots(figsize=( 10, 10))
trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
# plt.figure(figsize=(10, 7))
offset = 300 # offset the values by 300 timesteps 
plot_time_series(timesteps=X_test, values=y_test, start=offset, label="Test data")
# plot a line 'over' the dots (not exactly, off by 1)
plot_time_series(timesteps=X_test[1:], values=naive_forecast, format="-", 
                  start=offset, label="Model 0 Base: Naive forecast");
plt.text(.3, .2, horizontalalignment='right', transform=trans, va='bottom',
         s=rslts0, wrap=True,
         fontsize=14, bbox=dict(facecolor='pink', alpha=0.5))
plt.show()
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
HORIZON = 1 # predict 1 step at a time
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
#%% I will have to return to this sometime later.  Too hard now.
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/timeseries_dataset_from_array?hl=ko
# from tensorflow.keras.preprocessing import timeseries_datasest_from_array
ds = tf.keras.preprocessing.timeseries_dataset_from_array(prices, targets=None, 
                                                          sequence_length = 7)

#%% # View the first 3 windows/labels
for i in range(3):
  print(f"Window: {full_windows[i]} -> Label: {full_labels[i]}")
  
#%% DEF make_train_test_splits(windows, labels, test_split=0.2) 
def make_train_test_splits(windows, labels, test_split=0.2):
  """
  Splits matching pairs of windows and labels into train and test splits.
  """
  split_size = int(len(windows) * (1-test_split)) #default to 80% train/20% test
  train_windows = windows[:split_size]
  train_labels = labels[:split_size]
  test_windows = windows[split_size:]
  test_labels = labels[split_size:]
  return train_windows, test_windows, train_labels, test_labels  

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

#%% # Make the train/test splits
train_windows, test_windows, train_labels, test_labels =\
    make_train_test_splits(full_windows, full_labels)
print(f'Window lengths: {len(train_windows)}, {len(test_windows)}, {len(train_labels)}, {len(test_labels)}')
train_windows[:5], train_labels[:5]

# Check to see if same (accounting for horizon and window size)
np.array_equal(np.squeeze(train_labels[:-HORIZON-1]), y_train[WINDOW_SIZE:])

#%% MODEL #1 TF.dense 2 hidden layers
import tensorflow as tf
from tensorflow.keras import layers

# Set random seed for as reproducible results as possible
tf.random.set_seed(42)

# Construct model
model_1 = tf.keras.Sequential([
  layers.Dense(128, activation="relu"),
  # the output layer below:  we're predicting a number, so we don't need 'activation'
  layers.Dense(HORIZON, activation="linear") # linear activation == no activation
], name="model_1_dense") # give the model a name so we can save it

# Compile model
model_1.compile(loss="mae", # a regression loss function
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["mae"]) # not needed when the loss function is already MAE

# Fit model   (using the previous 7 timesteps to predict next day)
model_1.fit(x=train_windows, # train windows of 7 timesteps of Bitcoin prices
            y=train_labels, # horizon value of 1 
            epochs=100,
            verbose=1,
            batch_size=128,  #this can be big because the batch size of 7 days is small
            validation_data=(test_windows, test_labels),
            # create ModelCheckpoint callback to save best model
            callbacks=[create_model_checkpoint(model_name=model_1.name)]) 

#%% # Evaluate (BEST) model on test data
model_1.evaluate(test_windows, test_labels)

# Load in saved best performing model_1 and evaluate on test data
model_1 = tf.keras.models.load_model("model_experiments/model_1_dense")
model_1.evaluate(test_windows, test_labels)

#%% # Make predictions using model_1 on the test dataset and view the results
model_1_preds = make_preds(model_1, test_windows)
len(model_1_preds), model_1_preds[:10]

#%% # Evaluate preds
model_1_results = evaluate_preds(y_true=tf.squeeze(test_labels), # reduce to right shape
                                 y_pred=model_1_preds)
model_1_results

#%% MODEL 1 Plot
rslts1 = str(model_1_results).replace(',','\n')
fig, ax = plt.subplots(figsize=( 10, 10))
trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
offset = 300
plt.text(.1, .1, horizontalalignment='left', transform=trans, va='bottom',
         s=rslts1, wrap=True,
         fontsize=24, bbox=dict(facecolor='pink', alpha=0.5))
# Account for the test_window offset and index into test_labels to ensure correct plotting
plot_time_series(timesteps=X_test[-len(test_windows):], values=test_labels[:, 0], 
                 start=offset, label="Test_data")
plot_time_series(timesteps=X_test[-len(test_windows):], values=model_1_preds, 
                 start=offset, format="-", label="MDL1: tf.dense.2hidlayers window 7")
plt.show()
#%% MODEL 2 CREATE, compile, fit:  DENSE with window size of 30
HORIZON = 1 # predict one step at a time
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
model_2 = tf.keras.Sequential([
  layers.Dense(128, activation="relu"),
  layers.Dense(HORIZON) # need to predict horizon number of steps into the future
], name="model_2_dense")
model_2.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam())
model_2.fit(train_windows30,
            train_labels30,
            epochs=100,
            batch_size=128,
            verbose=0,
            validation_data=(test_windows30, test_labels30),
            callbacks=[create_model_checkpoint(model_name=model_2.name)])
#%% # Evaluate model 2 preds
model_2.evaluate(test_windows30, test_labels30)

# Load in best performing model
model_2 = tf.keras.models.load_model("model_experiments/model_2_dense/")
model_2.evaluate(test_windows30, test_labels30)

#%% # Get forecast predictions
model_2_preds = make_preds(model_2,
                           input_data=test_windows30)
# Evaluate results for model 2 predictions
model_2_results = evaluate_preds(y_true=tf.squeeze(test_labels30), # remove 1 dimension of test labels
                                 y_pred=model_2_preds)
model_2_results
#%% MODEL 2 chart... 
rslts2 = str(model_2_results).replace(',','\n')
fig, ax = plt.subplots(figsize=( 10, 10))
trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
offset = 300
#plt.figure(figsize=(10, 7))
# Account for the test_window offset
plot_time_series(timesteps=X_test[-len(test_windows30):], values=test_labels30[:, 0], 
                 start=offset, label="test_data")
plot_time_series(timesteps=X_test[-len(test_windows30):], values=model_2_preds, 
                 start=offset, format="-", label="MDL2: tf.dense.2hidlayers window 30") 
#ax.text(0.5, 0.5, "middle of graph", transform=trans)
plt.text(.1, .1, horizontalalignment='left', transform=trans, va='bottom',
         s=rslts2, wrap=True,
         fontsize=24, bbox=dict(facecolor='pink', alpha=0.5))
#%% ENDING
duration =  time.perf_counter() - starttime
if duration < 60:
    print('\nit  took: {:.2f} seconds\n'.format(duration))
    durationtm = str(round(duration,2)) + ' sec'
elif duration >= 60: 
    print('\nIt  took: {:.2f} minutes\n'.format(duration/60))
    duration = duration/60
    durationtm = str(duration) + ' min'

#%% Put the charts together for comparison...
#plt.figure(figsize=(12,6))
#plt.figure(figsize = (20,10), tight_layout=None)
datadesc = f'Training points: {len(X_train)} dates, {len(y_train)} prices.'\
    f' \n# of Testing Points: , {len(X_test)}, {len(y_test)}' \
        f'\nBut forecast is short 1 ie: {len(naive_forecast)}'
mdl1note1 = '''With these outputs, our model isn't forecasting yet. It's only making 
predictions on the test dataset. Forecasting would involve a model 
making predictions into the future, however, the test dataset is 
only a pseudofuture.'''
notes =  'first 3 Bitcoin Price windows:'
notes0 = f"\nWindow: {full_windows[0]} -> Label: {full_labels[0]}"
notes1 = f"\nWindow: {full_windows[1]} -> Label: {full_labels[1]}"
notes2 = f"\nWindow: {full_windows[2]} -> Label: {full_labels[2]}"
fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots\
   (2, 2, sharey=True, figsize = (20,20), sharex=True)
plt.subplots_adjust(wspace =.05, hspace = .08)  # https://www.machinelearningplus.com/plots/matplotlib-pyplot/
plt.suptitle(f'UDEMY TFCertZ2M {filename}   {modelstart}  Duration: {durationtm} '\
             '\nBitcoin price Actual & Forecast by '\
             'various models\nNotice shared AxisLabels, reduced margins'\
            ' &  Notice that the best was the NAIVE model!!!'\
                '\nSec 12 Lessons 288 thru 312'
             , fontsize=20, weight='bold')

#ax = fig.add_subplot(121)
plt.subplot(2,2,1)
trans1 = transforms.blended_transform_factory(ax1.transAxes, ax1.transAxes)
ax1.text(.3, .3, horizontalalignment='right', transform=trans1, 
         va='bottom', s=rslts0, wrap=True,
          fontsize=14, bbox=dict(facecolor='pink', alpha=0.5))
ax1.text(.03, .5, horizontalalignment='left', transform=trans1, 
         va='bottom', s= notes + notes0 + notes1 + notes2,
          fontsize=16, bbox=dict(facecolor='green', alpha=0.2))
ax1.text(.03, .73, horizontalalignment='left', transform=trans1, 
         va='bottom', s=datadesc, wrap=True,
          fontsize=14, bbox=dict(facecolor='blue', alpha=0.3))
plot_time_series_model(naive_results,  X_test[1:], 300, 
                       'Naive - (Orange Line)', 'MDL0:  np.round(y_test[:-1] ,2)',
                       naive_forecast)
plt.subplot(2,2,2)
trans2 = transforms.blended_transform_factory(ax2.transAxes, ax2.transAxes)
ax2.text(.3, .3, horizontalalignment='right', transform=trans2, 
         va='bottom', s=rslts1, wrap=True,
          fontsize=14, bbox=dict(facecolor='pink', alpha=0.5))
ax2.text(.03, .5, horizontalalignment='left', transform=trans2, 
         va='bottom', s=mdl1note1, wrap=True,
          fontsize=14, bbox=dict(facecolor='green', alpha=0.2))
plot_time_series_model(model_1_results,  X_test[-len(test_windows):], 300, 
                       'Dense - (Orange Line)', 'MDL1: tf.dense.2hidlayers window 7', model_1_preds)
plt.subplot(2,2,3)
trans3 = transforms.blended_transform_factory(ax3.transAxes, ax3.transAxes)
ax3.text(.3, .1, horizontalalignment='right', transform=trans3, 
         va='bottom', s=rslts2, wrap=True,
          fontsize=14, bbox=dict(facecolor='pink', alpha=0.5))
plot_time_series_model(model_2_results, X_test[-len(test_windows30):], 300, 
                       'Dense2', 'MDL2: tf.dense.2hidlayers window 30', model_2_preds)
# 4th SUBPLOT NOTES...
plt.subplot(2,2,4)
code ='''fig, ([ax1, ax2], [ax3, ax4]) = 
  plt.subplots(2, 2, sharey=True, figsize = (20,20), sharex=True)
plt.subplots_adjust(wspace =.05, hspace = .08)  
# https://www.machinelearningplus.com/plots/matplotlib-pyplot/
plt.suptitle(f'{filename}   {modelstart}\nBitcoin price Actual & Forecast by'''

code2 ='''#ax = fig.add_subplot(121)
plt.subplot(2,2,1)
trans1 = transforms.blended_transform_factory(ax1.transAxes, 
                                              ax1.transAxes)
ax1.text(.1, .3, horizontalalignment='left', transform=trans1, 
         va='bottom', s=rslts0, wrap=True,
          fontsize=14, bbox=dict(facecolor='white', alpha=0.5))'''
code3 = '''# Naïve forecast equals every value excluding the last value
\nnaive_forecast = y_test[:-1] '''

code4 ='''# Construct model
model_1 = tf.keras.Sequential([
  layers.Dense(128, activation="relu"),
  layers.Dense(HORIZON, activation="linear") # linear activation == no activation
], name="model_1_dense") # give the model a name so we can save it
# Compile model
model_1.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["mae"]) # not needed when the loss function is already MAE
# Fit model   (using the previous 7 timesteps to predict next day)
model_1.fit(x=train_windows, # train windows of 7 timesteps of Bitcoin prices
            y=train_labels, # horizon value of 1 
            epochs=100,
            verbose=1,
            batch_size=128,
            validation_data=(test_windows, test_labels),
            # create ModelCheckpoint callback to save best model
            callbacks=[create_model_checkpoint(model_name=model_1.name)]) '''
ax4.set_title('Key pieces of code')
trans4 = transforms.blended_transform_factory(ax4.transAxes, ax4.transAxes)
ax4.text(.1, .99, horizontalalignment='left', transform=trans4, 
         va='top', s=code, wrap=True,
          fontsize=14, bbox=dict(facecolor='white', alpha=0.5))
ax4.text(.1, .78, horizontalalignment='left', transform=trans4, 
         va='top', s=code2, wrap=True,
          fontsize=14, bbox=dict(facecolor='white', alpha=0.5))
ax4.text(.1, .51, horizontalalignment='left', transform=trans4, 
         va='top', s=code3, wrap=True,
          fontsize=14, bbox=dict(facecolor='white', alpha=0.5))
ax4.text(.1, .39, horizontalalignment='left', transform=trans4, 
         va='top', s=code4, wrap=True,
          fontsize=9, bbox=dict(facecolor='white', alpha=0.5))

plt.show()

