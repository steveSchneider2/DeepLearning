# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 17:06:58 2021

@author: steve
"""

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:39:31.204530Z","iopub.execute_input":"2021-06-14T19:39:31.205254Z","iopub.status.idle":"2021-06-14T19:39:31.215719Z","shell.execute_reply.started":"2021-06-14T19:39:31.205117Z","shell.execute_reply":"2021-06-14T19:39:31.214887Z"}}
import pandas as pd
import numpy as np
from datetime import timedelta
import gc
from functools import reduce

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:39:36.146784Z","iopub.execute_input":"2021-06-14T19:39:36.147421Z","iopub.status.idle":"2021-06-14T19:39:36.156571Z","shell.execute_reply.started":"2021-06-14T19:39:36.147362Z","shell.execute_reply":"2021-06-14T19:39:36.155596Z"}}
def make_df(df, col, bool_in=False):
    tp = df.loc[ ~df[col].isnull() ,[col]].copy()
    df.drop(col, axis=1, inplace=True)
    
    tp[col] = tp[col].str.replace("null",'""')
    if bool_in:
        tp[col] = tp[col].str.replace("false",'"False"')
        tp[col] = tp[col].str.replace("true",'"True"')
    tp[col] = tp[col].apply(lambda x: eval(x) )
    a = tp[col].sum()
    gc.collect()
    return pd.DataFrame(a)
#===============

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:39:37.977651Z","iopub.execute_input":"2021-06-14T19:39:37.978345Z","iopub.status.idle":"2021-06-14T19:39:37.982455Z","shell.execute_reply.started":"2021-06-14T19:39:37.978305Z","shell.execute_reply":"2021-06-14T19:39:37.981692Z"}}
# ROOT_DIR = "../input/mlb-player-digital-engagement-forecasting"
ROOT_DIR = "D:/data/mlb-player-digital-engagement"

# %% [markdown]
# ## UTILITY FUNCTIONS

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:39:43.458312Z","iopub.execute_input":"2021-06-14T19:39:43.458729Z","iopub.status.idle":"2021-06-14T19:39:43.466654Z","shell.execute_reply.started":"2021-06-14T19:39:43.458690Z","shell.execute_reply":"2021-06-14T19:39:43.465325Z"}}
#=======================#
def flatten(df, col):
    du = (df.pivot(index="playerId", columns="EvalDate", 
               values=col).add_prefix(f"{col}_").
      rename_axis(None, axis=1).reset_index())
    return du
#============================#
def reducer(left, right):
    return left.merge(right, on="playerId")
#========================

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:39:46.365550Z","iopub.execute_input":"2021-06-14T19:39:46.365913Z","iopub.status.idle":"2021-06-14T19:39:46.376466Z","shell.execute_reply.started":"2021-06-14T19:39:46.365882Z","shell.execute_reply":"2021-06-14T19:39:46.375445Z"}}
TGTCOLS = ["target1","target2","target3","target4"]
def train_lag(df, lag=1):
    dp = df[["playerId","EvalDate"]+TGTCOLS].copy()
    dp["EvalDate"]  =dp["EvalDate"] + timedelta(days=lag) 
    df = df.merge(dp, on=["playerId", "EvalDate"], suffixes=["",f"_{lag}"], how="left")
    return df
#=================================
def test_lag(sub):
    sub["playerId"] = sub["date_playerId"].apply(lambda s: int(  s.split("_")[1]  ) )
    assert sub.date.nunique() == 1
    dte = sub["date"].unique()[0]
    
    eval_dt = pd.to_datetime(dte, format="%Y%m%d")
    dtes = [eval_dt + timedelta(days=-k) for k in LAGS]
    mp_dtes = {eval_dt + timedelta(days=-k):k for k in LAGS}
    
    sl = LAST.loc[LAST.EvalDate.between(dtes[-1], dtes[0]), ["EvalDate","playerId"]+TGTCOLS].copy()
    sl["EvalDate"] = sl["EvalDate"].map(mp_dtes)
    du = [flatten(sl, col) for col in TGTCOLS]
    du = reduce(reducer, du)
    return du, eval_dt
    #
#===============

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:39:50.888153Z","iopub.execute_input":"2021-06-14T19:39:50.888609Z","iopub.status.idle":"2021-06-14T19:39:55.242100Z","shell.execute_reply.started":"2021-06-14T19:39:50.888571Z","shell.execute_reply":"2021-06-14T19:39:55.240657Z"}}
#%%%time
tr = pd.read_csv(f"{ROOT_DIR}/train.csv",  nrows=1100)
#tr = pd.read_csv(f"{ROOT_DIR}/train.csv", iterator=True, chunksize=10000)
#tr = pd.read_csv("../input/mlb-data/target.csv")
print(tr.shape)
gc.collect()

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:39:57.258362Z","iopub.execute_input":"2021-06-14T19:39:57.258799Z","iopub.status.idle":"2021-06-14T19:39:58.029745Z","shell.execute_reply.started":"2021-06-14T19:39:57.258766Z","shell.execute_reply":"2021-06-14T19:39:58.028541Z"}}
tr["EvalDate"] = pd.to_datetime(tr["EvalDate"])
tr["EvalDate"] = tr["EvalDate"] + timedelta(days=-1)
tr["EvalYear"] = tr["EvalDate"].dt.year

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:40:00.249363Z","iopub.execute_input":"2021-06-14T19:40:00.249745Z","iopub.status.idle":"2021-06-14T19:40:00.750591Z","shell.execute_reply.started":"2021-06-14T19:40:00.249710Z","shell.execute_reply":"2021-06-14T19:40:00.749414Z"}}
MED_DF = tr.groupby(["playerId","EvalYear"])[TGTCOLS].median().reset_index()
MEDCOLS = ["tgt1_med","tgt2_med", "tgt3_med", "tgt4_med"]
MED_DF.columns = ["playerId","EvalYear"] + MEDCOLS

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:40:02.955300Z","iopub.execute_input":"2021-06-14T19:40:02.955670Z","iopub.status.idle":"2021-06-14T19:40:02.978038Z","shell.execute_reply.started":"2021-06-14T19:40:02.955641Z","shell.execute_reply":"2021-06-14T19:40:02.976669Z"}}
MED_DF.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:40:08.062408Z","iopub.execute_input":"2021-06-14T19:40:08.062816Z","iopub.status.idle":"2021-06-14T19:40:08.068341Z","shell.execute_reply.started":"2021-06-14T19:40:08.062781Z","shell.execute_reply":"2021-06-14T19:40:08.066872Z"}}
LAGS = [1,2,3]
FECOLS = [f"{col}_{lag}" for lag in reversed(LAGS) for col in TGTCOLS]

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:40:09.744373Z","iopub.execute_input":"2021-06-14T19:40:09.744780Z","iopub.status.idle":"2021-06-14T19:40:13.878189Z","shell.execute_reply.started":"2021-06-14T19:40:09.744744Z","shell.execute_reply":"2021-06-14T19:40:13.876960Z"}}
%%time
for lag in LAGS:
    tr = train_lag(tr, lag=lag)
#===========

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:40:13.879970Z","iopub.execute_input":"2021-06-14T19:40:13.880315Z","iopub.status.idle":"2021-06-14T19:40:15.772787Z","shell.execute_reply.started":"2021-06-14T19:40:13.880285Z","shell.execute_reply":"2021-06-14T19:40:15.771633Z"}}
tr = tr.sort_values(by=["playerId", "EvalDate"])

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:40:19.717350Z","iopub.execute_input":"2021-06-14T19:40:19.717761Z","iopub.status.idle":"2021-06-14T19:40:20.137543Z","shell.execute_reply.started":"2021-06-14T19:40:19.717727Z","shell.execute_reply":"2021-06-14T19:40:20.128933Z"}}
print(tr.shape)
tr = tr.dropna()
print(tr.shape)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:40:21.679975Z","iopub.execute_input":"2021-06-14T19:40:21.680436Z","iopub.status.idle":"2021-06-14T19:40:22.294440Z","shell.execute_reply.started":"2021-06-14T19:40:21.680399Z","shell.execute_reply":"2021-06-14T19:40:22.293294Z"}}
tr = tr.merge(MED_DF, on=["playerId","EvalYear"])

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:40:23.266347Z","iopub.execute_input":"2021-06-14T19:40:23.266757Z","iopub.status.idle":"2021-06-14T19:40:23.297088Z","shell.execute_reply.started":"2021-06-14T19:40:23.266724Z","shell.execute_reply":"2021-06-14T19:40:23.295898Z"}}
tr.head(1)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:40:27.349006Z","iopub.execute_input":"2021-06-14T19:40:27.349387Z","iopub.status.idle":"2021-06-14T19:40:27.664553Z","shell.execute_reply.started":"2021-06-14T19:40:27.349357Z","shell.execute_reply":"2021-06-14T19:40:27.663490Z"}}
X = tr[FECOLS+MEDCOLS].values
y = tr[TGTCOLS].values

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:40:28.643902Z","iopub.execute_input":"2021-06-14T19:40:28.644292Z","iopub.status.idle":"2021-06-14T19:40:28.650246Z","shell.execute_reply.started":"2021-06-14T19:40:28.644257Z","shell.execute_reply":"2021-06-14T19:40:28.649180Z"}}
X.shape

# %% [markdown]
# ## Neural Net Training

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:40:50.910034Z","iopub.execute_input":"2021-06-14T19:40:50.910432Z","iopub.status.idle":"2021-06-14T19:40:58.226280Z","shell.execute_reply.started":"2021-06-14T19:40:50.910396Z","shell.execute_reply":"2021-06-14T19:40:58.225331Z"}}
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from sklearn.metrics import mean_absolute_error, mean_squared_error

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:40:58.227547Z","iopub.execute_input":"2021-06-14T19:40:58.227966Z","iopub.status.idle":"2021-06-14T19:40:58.234006Z","shell.execute_reply.started":"2021-06-14T19:40:58.227936Z","shell.execute_reply":"2021-06-14T19:40:58.233245Z"}}
def make_model(n_in):
    inp = L.Input(name="inputs", shape=(n_in,))
    x = L.Dense(50, activation="relu", name="d1")(inp)
    x = L.Dense(50, activation="relu", name="d2")(x)
    preds = L.Dense(4, activation="linear", name="preds")(x)
    
    model = M.Model(inp, preds, name="ANN")
    model.compile(loss="mean_absolute_error", optimizer="adam")
    return model

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:41:00.475964Z","iopub.execute_input":"2021-06-14T19:41:00.476481Z","iopub.status.idle":"2021-06-14T19:41:00.589956Z","shell.execute_reply.started":"2021-06-14T19:41:00.476449Z","shell.execute_reply":"2021-06-14T19:41:00.589007Z"}}
net = make_model(X.shape[1])
print(net.summary())

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:41:08.720497Z","iopub.execute_input":"2021-06-14T19:41:08.720932Z","iopub.status.idle":"2021-06-14T19:41:32.157036Z","shell.execute_reply.started":"2021-06-14T19:41:08.720896Z","shell.execute_reply":"2021-06-14T19:41:32.156127Z"}}
reg = make_model(X.shape[1])
reg.fit(X, y, epochs=10, batch_size=30_000)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:42:05.439580Z","iopub.execute_input":"2021-06-14T19:42:05.440225Z","iopub.status.idle":"2021-06-14T19:42:28.124818Z","shell.execute_reply.started":"2021-06-14T19:42:05.440180Z","shell.execute_reply":"2021-06-14T19:42:28.123622Z"}}
reg.fit(X, y, epochs=10, batch_size=30_000)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:42:35.945156Z","iopub.execute_input":"2021-06-14T19:42:35.945689Z","iopub.status.idle":"2021-06-14T19:42:37.149107Z","shell.execute_reply.started":"2021-06-14T19:42:35.945644Z","shell.execute_reply":"2021-06-14T19:42:37.147917Z"}}
pred = reg.predict(X, batch_size=50_000, verbose=1)
mae = mean_absolute_error(y, pred)
mse = mean_squared_error(y, pred, squared=False)
print("mae:", mae)
print("mse:", mse)

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:42:54.561879Z","iopub.execute_input":"2021-06-14T19:42:54.562291Z","iopub.status.idle":"2021-06-14T19:42:54.620499Z","shell.execute_reply.started":"2021-06-14T19:42:54.562254Z","shell.execute_reply":"2021-06-14T19:42:54.619336Z"}}
# Historical information to use in prediction time
bound_dt = pd.to_datetime("2021-01-01")
LAST = tr.loc[tr.EvalDate>bound_dt].copy()

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:42:57.240646Z","iopub.execute_input":"2021-06-14T19:42:57.241268Z","iopub.status.idle":"2021-06-14T19:42:57.249619Z","shell.execute_reply.started":"2021-06-14T19:42:57.241228Z","shell.execute_reply":"2021-06-14T19:42:57.248356Z"}}
LAST_MED_DF = MED_DF.loc[MED_DF.EvalYear==2021].copy()
LAST_MED_DF.drop("EvalYear", axis=1, inplace=True)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:42:59.978438Z","iopub.execute_input":"2021-06-14T19:42:59.979182Z","iopub.status.idle":"2021-06-14T19:42:59.986899Z","shell.execute_reply.started":"2021-06-14T19:42:59.979129Z","shell.execute_reply":"2021-06-14T19:42:59.985882Z"}}
LAST.shape, LAST_MED_DF.shape, MED_DF.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:43:40.862861Z","iopub.execute_input":"2021-06-14T19:43:40.863269Z","iopub.status.idle":"2021-06-14T19:43:41.281587Z","shell.execute_reply.started":"2021-06-14T19:43:40.863236Z","shell.execute_reply":"2021-06-14T19:43:41.279750Z"}}
#"""
import mlb
FE = []; SUB = [];
env = mlb.make_env() # initialize the environment
iter_test = env.iter_test() # iterator which loops over each date in test set

for (test_df, sub) in iter_test:
    # Features computation at Evaluation Date
    sub = sub.reset_index()
    sub_fe, eval_dt = test_lag(sub)
    sub_fe = sub_fe.merge(LAST_MED_DF, on="playerId", how="left")
    sub_fe = sub_fe.fillna(0.)
    
    _preds = reg.predict(sub_fe[FECOLS + MEDCOLS])
    sub_fe[TGTCOLS] = np.clip(_preds, 0, 100)
    sub.drop(["date"]+TGTCOLS, axis=1, inplace=True)
    sub = sub.merge(sub_fe[["playerId"]+TGTCOLS], on="playerId", how="left")
    sub.drop("playerId", axis=1, inplace=True)
    sub = sub.fillna(0.)
    # Submit
    env.predict(sub)
    # Update Available information
    sub_fe["EvalDate"] = eval_dt
    #sub_fe.drop(MEDCOLS, axis=1, inplace=True)
    LAST = LAST.append(sub_fe)
    LAST = LAST.drop_duplicates(subset=["EvalDate","playerId"], keep="last")
#"""

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:43:50.562409Z","iopub.execute_input":"2021-06-14T19:43:50.562966Z","iopub.status.idle":"2021-06-14T19:43:50.578982Z","shell.execute_reply.started":"2021-06-14T19:43:50.562920Z","shell.execute_reply":"2021-06-14T19:43:50.577725Z"}}
sub.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T19:43:55.528520Z","iopub.execute_input":"2021-06-14T19:43:55.528889Z","iopub.status.idle":"2021-06-14T19:43:55.534793Z","shell.execute_reply.started":"2021-06-14T19:43:55.528859Z","shell.execute_reply":"2021-06-14T19:43:55.533808Z"}}
tr.shape, LAST.shape, sub_fe.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-06-14T09:50:49.268983Z","iopub.execute_input":"2021-06-14T09:50:49.269332Z","iopub.status.idle":"2021-06-14T09:50:49.279831Z","shell.execute_reply.started":"2021-06-14T09:50:49.269304Z","shell.execute_reply":"2021-06-14T09:50:49.278753Z"}}
#df_tr["dte"] = pd.to_datetime(df_tr["date"], format='%Y%m%d')