"""
Created on Wed Aug 25 13:54:44 2021
@author: steve

PLAN:  This is part 3 of 12BitcoinTimeSeries.py

Model 
Num	Model       Type	Horizon Window 	Extra data
0	Naïve model (baseline)	NA	NA	NA
1	tf.Dense model            1	7	NA
2	Same as 1	              1	30	NA
3	Same as 1	              7	30	NA
4	Conv1D	                  1	7	NA
5	LSTM	                  1	7	NA

** starting here...
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

# Block reward values
block_reward_1 = 50 # 3 January 2009 (2009-01-03) - this block reward isn't in our dataset (it starts from 01 October 2013)
block_reward_2 = 25 # 28 November 2012 
block_reward_3 = 12.5 # 9 July 2016
block_reward_4 = 6.25 # 11 May 2020

# Block reward dates (datetime form of the above date stamps)
block_reward_2_datetime = np.datetime64("2012-11-28")
block_reward_3_datetime = np.datetime64("2016-07-09")
block_reward_4_datetime = np.datetime64("2020-05-11")

# Get date indexes for when to add in different block dates
block_reward_2_days = (block_reward_3_datetime - bitcoin_prices.index[0]).days
block_reward_3_days = (block_reward_4_datetime - bitcoin_prices.index[0]).days
block_reward_2_days, block_reward_3_days

# Add block_reward column
bitcoin_prices_block = bitcoin_prices.copy()
bitcoin_prices_block["block_reward"] = None

# Set values of block_reward column (it's the last column hence -1 indexing on iloc)
bitcoin_prices_block.iloc[:block_reward_2_days, -1] = block_reward_2
bitcoin_prices_block.iloc[block_reward_2_days:block_reward_3_days, -1] = block_reward_3
bitcoin_prices_block.iloc[block_reward_3_days:, -1] = block_reward_4
bitcoin_prices_block

# Plot the block reward/price over time
# Note: Because of the different scales of our values we'll scale them to be between 0 and 1.
from sklearn.preprocessing import minmax_scale
scaled_price_block_df = pd.DataFrame(minmax_scale(bitcoin_prices_block[["Price", "block_reward"]]), # we need to scale the data first
                                     columns=bitcoin_prices_block.columns,
                                     index=bitcoin_prices_block.index)
scaled_price_block_df.plot(figsize=(10, 7));

#%%  Making a windowed dataset with pandas
# Previously, we used some custom made functions to window our univariate time series.
# However, since we've just added another variable to our dataset, these functions won't work.
# Not to worry though. Since our data is in a pandas DataFrame, we can leverage the pandas.DataFrame.shift() method to create a windowed multivariate time series.
# The shift() method offsets an index by a specified number of periods.

# Setup dataset hyperparameters
HORIZON = 1
WINDOW_SIZE = 7

# Make a copy of the Bitcoin historical data with block reward feature
bitcoin_prices_windowed = bitcoin_prices_block.copy()
# Add windowed columns
for i in range(WINDOW_SIZE): # Shift values for each step in WINDOW_SIZE
  bitcoin_prices_windowed[f"Price+{i+1}"] = bitcoin_prices_windowed["Price"].shift(periods=i+1)
bitcoin_prices_windowed.head(20)

# We'll also remove the NaN values using pandas dropna() method, this equivalent to starting our windowing function at sample 0 (the first sample) + WINDOW_SIZE.
# Let's create X & y, remove the NaN's and convert to float32 to prevent TensorFlow errors 
X = bitcoin_prices_windowed.dropna().drop("Price", axis=1).astype(np.float32) 
y = bitcoin_prices_windowed.dropna()["Price"].astype(np.float32)
X.head()
y.head()

#%% Create train and test splits the right way for time series data
split_size = int(0.8 * len(X)) # 80% train, 20% test

# Create train data splits (everything before the split)
X_train, y_train = X[:split_size], y[:split_size]

# Create test data splits (everything after the split)
X_test, y_test = X[split_size:], y[split_size:]

len(X_train), len(X_test), len(y_train), len(y_test)
print(f'Training points: {len(X_train)} dates, {len(y_train)} prices. \n# of Testing Points: , {len(X_test)}, {len(y_test)}')
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
#%% MODEL #6 TF.dense 2 hidden layers
import tensorflow as tf
from tensorflow.keras import layers

# Set random seed for as reproducible results as possible
tf.random.set_seed(42)
mdl6start = time.perf_counter()

# Construct model
model_6 = tf.keras.Sequential([
  layers.Dense(128, activation="relu"),
  layers.Dense(128, activation="relu"),
  # the output layer below:  we're predicting a number, so we don't need 'activation'
  layers.Dense(HORIZON, activation="linear") # linear activation == no activation
], name="mdl_6_DenseMultiVar") # give the model a name so we can save it

# Compile model
model_6.compile(loss="mae", # a regression loss function
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["mae"]) # not needed when the loss function is already MAE

# Fit model   (using the previous 7 timesteps to predict next day)
model_6.fit(x=X_train, # train windows of 7 timesteps of Bitcoin prices
            y=y_train, # horizon value of 1 
            epochs=100,
            verbose=1,
            batch_size=128,  #this can be big because the batch size of 7 days is small
            validation_data=(X_test, y_test),
            # create ModelCheckpoint callback to save best model
            callbacks=[create_model_checkpoint(model_name=model_6.name)]) 
mdl6duration = time.perf_counter() - mdl6start
model_6.summary()
#%% # Evaluate model 6 preds
model_6.predict
model_6.evaluate(X_test, y_test)

# Load in best performing model
model_6 = tf.keras.models.load_model("model_experiments/mdl_6_DenseMultiVar/")
model_6.evaluate(X_test, y_test)

#%% # Get forecast predictions
model_6_preds = make_preds(model_6, input_data=X_test)
# Evaluate results for model 6 predictions
model_6_results = evaluate_preds(y_true=tf.squeeze(y_test), # remove 1 dimension of test labels
                                 y_pred=model_6_preds)
model_6_results
#%% MODEL 6 chart... 
note61 = 'Dense Layer is also called fully connected layer.\n\n'\
'Still trying to beat Naive model, we added the "halving" value for this prediction.'\
    '\nIt DID make an improvement.  Not at all sure why!'
rslts6 = str(model_6_results).replace(',','\n')
fig, ax = plt.subplots(figsize=( 10, 10))
trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
offset = 300
plot_time_series(timesteps=X_test.index, 
                 values=y_test.values, 
                 start=offset, label="test_data")
plot_time_series(timesteps=X_test.index, 
                  values=model_6_preds,
                  start=offset, format="-", 
                  label=f'MDL6: tf.3hidlayers Horizon: {HORIZON} window {WINDOW_SIZE}') 
plt.title(f'MULTIVARIATE! {modelstart}   Duration: {round(mdl6duration,2)} sec' \
          f'\nBitcoin prices + "Halving" value;            {filename}')
ax.text(0.05, 0.65, tps.mdl2txt(model_6), transform=trans)
ax.text(0.05, 0.35, note61, transform=trans)
plt.text(.3, .1, horizontalalignment='right', transform=trans, va='bottom',
         s=rslts6, wrap=True,
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

# datadesc = f'Training points: {len(X_train)} dates, {len(y_train)} prices.'\
#     f' \n# of Testing Points: , {len(X_test)}, {len(y_test)}' \
#         f'\nBut forecast is short 1 ie: {len(model_5_preds)}'
# mdl1note1 = '''With these outputs, our model isn't forecasting yet. It's only making 
# predictions on the test dataset. Forecasting would involve a model 
# making predictions into the future, however, the test dataset is 
# only a pseudofuture.'''
# notes =  'first 3 Bitcoin Price windows:'
# notes0 = f"\nWindow: {full_windows[0]} -> Label: {full_labels[0]}"
# notes1 = f"\nWindow: {full_windows[1]} -> Label: {full_labels[1]}"
# notes2 = f"\nWindow: {full_windows[2]} -> Label: {full_labels[2]}"
# # https://www.machinelearningplus.com/plots/matplotlib-pyplot/
# fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots\
#    (2, 2, sharey=True, figsize = (20,20), sharex=True)
# plt.subplots_adjust(wspace =.05, hspace = .08)  
# plt.suptitle(f'UDEMY TFCertZ2M {filename}   {modelstart}  Duration: {durationtm} '\
#              '\nBitcoin price Actual & Forecast by '\
#              'various models\nNotice shared AxisLabels, reduced margins'\
#             ' &  Notice that the best was the NAIVE model!!!'\
#                 '\nSec 12 Lessons 313 thru 319'
#              , fontsize=20, weight='bold')

# #ax = fig.add_subplot(121)
# plt.subplot(2,2,1)
# trans1 = transforms.blended_transform_factory(ax1.transAxes, ax1.transAxes)
# ax1.text(.3, .3, horizontalalignment='right', transform=trans1, 
#          va='bottom', s=rslts3, wrap=True,
#           fontsize=14, bbox=dict(facecolor='pink', alpha=0.5))
# ax1.text(.03, .5, horizontalalignment='left', transform=trans1, 
#          va='bottom', s= notes + notes0 + notes1 + notes2,
#          fontsize=16, bbox=dict(facecolor='green', alpha=0.2))
# ax1.text(.03, .73, horizontalalignment='left', transform=trans1, 
#          va='bottom', s=datadesc, wrap=True,
#           fontsize=14, bbox=dict(facecolor='blue', alpha=0.3))
# ax1.text(.23, .03, horizontalalignment='left', transform=trans1, 
#          va='bottom', s=tps.mdl2txt(model_3), wrap=True,
#           fontsize=14, bbox=dict(facecolor='blue', alpha=0.3))
# plot_time_series_model(model_3_results,  X_test[-len(test_windows30):], 300, 
#                        f'{model_3.name} - (Orange Line)',
#                        f'Mdl3: WindowSize 30 Horizon: 7  ',
#                        tf.reduce_mean(model_3_preds, axis=1))
# plt.subplot(2,2,2)
# trans2 = transforms.blended_transform_factory(ax2.transAxes, ax2.transAxes)
# ax2.text(.3, .3, horizontalalignment='right', transform=trans2, 
#          va='bottom', s=rslts4, wrap=True,
#           fontsize=14, bbox=dict(facecolor='pink', alpha=0.5))
# # ax1.text(.03, .5, horizontalalignment='left', transform=trans1, 
# #          va='bottom', s= notes + notes0 + notes1 + notes2,
# #           fontsize=16, bbox=dict(facecolor='green', alpha=0.2))
# ax2.text(.03, .73, horizontalalignment='left', transform=trans2, 
#          va='bottom', s=datadesc, wrap=True,
#           fontsize=14, bbox=dict(facecolor='blue', alpha=0.3))
# ax2.text(.23, .03, horizontalalignment='left', transform=trans2, 
#          va='bottom', s=tps.mdl2txt(model_4), wrap=True,
#           fontsize=14, bbox=dict(facecolor='blue', alpha=0.3))
# plot_time_series_model(model_4_results,  X_test[-len(test_windows):], 300, 
#                        f'{model_4.name}- (Orange Line)  Time: {round(mdl4duration,2)} sec', 
#                        f'MDL4: Conv1d  WindowSize: {WINDOW_SIZE}; Horizon: {HORIZON}',
#                        model_4_preds)
# plt.subplot(2,2,3)
# trans3 = transforms.blended_transform_factory(ax3.transAxes, ax3.transAxes)
# ax3.text(.3, .3, horizontalalignment='right', transform=trans3, 
#          va='bottom', s=rslts5, wrap=True,
#           fontsize=14, bbox=dict(facecolor='pink', alpha=0.5))
# ax3.text(.03, .5, horizontalalignment='left', transform=trans3, 
#          va='bottom', s=mdl1note1, wrap=True,
#           fontsize=14, bbox=dict(facecolor='green', alpha=0.2))
# ax3.text(.3, .03, horizontalalignment='left', transform=trans3, 
#          va='bottom', s=tps.mdl2txt(model_5), wrap=True,
#           fontsize=14, bbox=dict(facecolor='blue', alpha=0.3))
# plot_time_series_model(model_5_results,  X_test[-len(test_windows):], 300, 
#                        f'{model_5.name}:(Orange Line)   Time: {round(mdl5duration,2)} sec', 
#                        f'MDL5: tf.LSTM.2hidlayers WindowSize: {WINDOW_SIZE}; Horizon: {HORIZON}', 
#                        model_5_preds)

# plt.show()
# #%%

