# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 17:06:02 2021

@author: steve
"""

#%% starters
import sys, time
path = 'C:/Users/steve/Documents/GitHub/misc'
sys.path
sys.path.insert(0,path)
import tensorPrepStarter as tps

starttime, modelstart, filename = tps.setstartingconditions()
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
bitcoin_prices.plot(figsize=(10, 7))
plt.ylabel("BTC Price")
plt.title("Price of Bitcoin from 1 Oct 2013 to 18 May 2021", fontsize=16)
plt.legend(fontsize=14);

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
plt.figure(figsize=(10, 7))
plt.plot(timesteps, btc_price)
plt.title("Price of Bitcoin from 1 Oct 2013 to 18 May 2021", fontsize=16)
plt.xlabel("Date")
#plt.legend(fontsize=14)
plt.ylabel("BTC Price");

#%% Get bitcoin date array
timesteps = bitcoin_prices.index.to_numpy()
prices = bitcoin_prices["Price"].to_numpy()

timesteps[:10], prices[:10]

#%% Create train and test splits the right way for time series data
split_size = int(0.8 * len(prices)) # 80% train, 20% test

# Create train data splits (everything before the split)
X_train, y_train = timesteps[:split_size], prices[:split_size]

# Create test data splits (everything after the split)
X_test, y_test = timesteps[split_size:], prices[split_size:]

len(X_train), len(X_test), len(y_train), len(y_test)

#%% Plot correctly made splits
plt.figure(figsize=(10, 7))
plt.scatter(X_train, y_train, s=5, color='blue', label="Train data")
plt.scatter(X_test, y_test, s=5, color='red', label="Test data")
plt.xlabel("Date")
plt.ylabel("BTC Price")
plt.legend(fontsize=14)
plt.show();

#%% Create a function to plot time series data
def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
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
plt.figure(figsize=(10, 7))
plot_time_series(timesteps=X_train, values=y_train, label="Train data")
plot_time_series(timesteps=X_test, values=y_test, label="Test data")
#%% # Create a naïve forecast
naive_forecast = y_test[:-1] # Naïve forecast equals every value excluding the last value
naive_forecast[:10], naive_forecast[-10:] # View frist 10 and last 10 

#%% # Plot naive forecast
plt.figure(figsize=(10, 7))
plot_time_series(timesteps=X_train, values=y_train, label="Train data")
plot_time_series(timesteps=X_test, values=y_test, label="Test data")
plot_time_series(timesteps=X_test[1:], values=naive_forecast, format="-", label="Naive forecast");

#%% ENDING
duration =  time.perf_counter() - starttime
if duration < 60:
    print('\nit  took: {:.2f} seconds\n'.format(duration))
elif duration >= 60: 
    print('\nIt  took: {:.2f} minutes\n'.format(duration/60))
    duration = duration/60
    durationtm = str(duration) + ' min'
