# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 20:26:43 2021

@author: steve
"""

# %% [markdown] {"execution":{"iopub.execute_input":"2021-06-04T23:07:54.351665Z","iopub.status.busy":"2021-06-04T23:07:54.351302Z","iopub.status.idle":"2021-06-04T23:07:54.356206Z","shell.execute_reply":"2021-06-04T23:07:54.355039Z","shell.execute_reply.started":"2021-06-04T23:07:54.351637Z"}}
# ## About this
# * Notebook for creating [unnested data sets](https://www.kaggle.com/naotaka1128/mlb-unnested).
# * I made this based on [official notebook](https://www.kaggle.com/alokpattani/mlb-player-digital-engagement-data-exploration).
# * I saved as a pickle file after reducing memory, if you want to use csv, please fork this notebook.
# 
# 
# ## Notebook Setup

# %% [code] {"_kg_hide-input":false,"execution":{"iopub.status.busy":"2021-06-11T10:35:27.126453Z","iopub.execute_input":"2021-06-11T10:35:27.126955Z","iopub.status.idle":"2021-06-11T10:35:27.169815Z","shell.execute_reply.started":"2021-06-11T10:35:27.12684Z","shell.execute_reply":"2021-06-11T10:35:27.168718Z"}}
#### Import Python Libraries and Set Script Options ####
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR = "D:/data/mlb-player-digital-engagement"

# Lists all input data files from "../input/" directory
import os
for dirname, _, filenames in os.walk(ROOT_DIR):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% [code] {"execution":{"iopub.status.busy":"2021-06-11T10:35:27.173463Z","iopub.execute_input":"2021-06-11T10:35:27.173807Z","iopub.status.idle":"2021-06-11T10:35:27.1894Z","shell.execute_reply.started":"2021-06-11T10:35:27.173774Z","shell.execute_reply":"2021-06-11T10:35:27.188016Z"}}
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float64)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# %% [markdown]
# ## Load Data

# %% [code] {"execution":{"iopub.status.busy":"2021-06-11T10:37:07.368254Z","iopub.execute_input":"2021-06-11T10:37:07.368642Z","iopub.status.idle":"2021-06-11T10:37:07.566525Z","shell.execute_reply.started":"2021-06-11T10:37:07.36861Z","shell.execute_reply":"2021-06-11T10:37:07.565392Z"}}
# Start with input file path
input_file_path = Path(ROOT_DIR)

files = ['seasons', 'teams', 'players', 'awards', 'example_sample_submission']

for file in files:
    df = pd.read_csv(input_file_path / f"{file}.csv")
    #print(f"{file}.pickle")
    #display(df.head(3))
    reduce_mem_usage(df).to_pickle(f"{file}.pickle", protocol=4)
    #print('\n'*2)


# %% [code] {"_kg_hide-input":false,"execution":{"iopub.status.busy":"2021-06-11T10:35:27.444002Z","iopub.execute_input":"2021-06-11T10:35:27.444294Z","iopub.status.idle":"2021-06-11T10:35:40.596075Z","shell.execute_reply.started":"2021-06-11T10:35:27.444265Z","shell.execute_reply":"2021-06-11T10:35:40.593754Z"}}
for file in ['example_test', 'train']:
    # drop playerTwitterFollowers, teamTwitterFollowers from example_test
    df = pd.read_csv(input_file_path / f"{file}.csv").dropna(axis=1,how='all')
    daily_data_nested_df_names = df.drop('date', axis = 1).columns.values.tolist()

    for df_name in daily_data_nested_df_names:
        date_nested_table = df[['date', df_name]]

        date_nested_table = (date_nested_table[
          ~pd.isna(date_nested_table[df_name])
          ].
          reset_index(drop = True)
          )

        daily_dfs_collection = []

        for date_index, date_row in date_nested_table.iterrows():
            daily_df = pd.read_json(date_row[df_name])

            daily_df['dailyDataDate'] = date_row['date']

            daily_dfs_collection = daily_dfs_collection + [daily_df]

        # Concatenate all daily dfs into single df for each row
        unnested_table = (pd.concat(daily_dfs_collection,
          ignore_index = True).
          # Set and reset index to move 'dailyDataDate' to front of df
          set_index('dailyDataDate').
          reset_index()
          )
        #print(f"{file}_{df_name}.pickle")
        #display(unnested_table.head(3))
        reduce_mem_usage(unnested_table).to_pickle(f"{file}_{df_name}.pickle")
        #print('\n'*2)

        # Clean up tables and collection of daily data frames for this df
        del(date_nested_table, daily_dfs_collection, unnested_table)
# %% [code]
import pickle
infile = open('train_events.pickle','rb')
new_dict = pickle.load(infile)
infile.close()
