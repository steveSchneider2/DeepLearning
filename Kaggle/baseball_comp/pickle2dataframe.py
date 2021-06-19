# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 20:42:47 2021

@author: steve

https://www.datacamp.com/community/tutorials/pickle-python-tutorial
"""
from __future__ import print_function

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option('display.max_columns', 80)
pd.set_option('display.width', 140)
dftstGms.describe
#%% def total_size (o, handlers={}, verbose=False):
# https://code.activestate.com/recipes/577504/  Raymond Hettinger in 2010
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)
#%% List files
ROOT_DIR = "D:/data/mlb-player-digital-engagement"

# Lists all input data files from "../input/" directory
import os
for dirname, _, filenames in os.walk(ROOT_DIR):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#%% showme()
def showme(df):
    # type(df)
    # df.memory_usage()
    # df.columns
    df.info()
    # df.describe
    print(f'Shape:{df.shape} TtlSize: {total_size(df)} ')
    
#%% 
infile = open(ROOT_DIR+'/pickles/'+'train_events.pickle','rb')
dftrn_events = pickle.load(infile)  # brings in as a dataframe
infile.close()
showme(dftrn_events)
dftrn_events.describe


extestgms = open(ROOT_DIR+'/pickles/example_test_games.pickle','rb')
dftstGms = pickle.load(extestgms)  # brings in as a dataframe
extestgms.close()
showme(dftstGms)

exsubmit = open(ROOT_DIR+'/pickles/example_sample_submission.pickle','rb')
dfexsubm = pickle.load(exsubmit)  # brings in as a dataframe
exsubmit.close()
showme(dfexsubm)
dfexsubm.describe

type(dftstGms)
dftstGms.memory_usage()
dftstGms.columns
total_size(dftstGms)
dftstGms.shape
dftstGms.info()

new_dict.describe()
list(new_dict.items())

df = pd.DataFrame(list(new_dict.items())  ,columns = ['column1','column2','col3','col4']) 

df.memory_usage()
df.head()

import sys
dict(list(new_dict.items() )[:2])
sys.getsizeof(new_dict)


dict(list(new_dict.items() )[:4])



##### Example call #####
#%%
# if __name__ == '__main__':
#     d = dict(a=1, b=2, c=3, d=[4,5,6,7], e='a string of chars')
print(total_size(new_dict, verbose=True))
type(new_dict)
#%%
dict = {'Name' : ['Martha', 'Tim', 'Rob', 'Georgia'],
        'Maths' : [87, 91, 97, 95],
        'Science' : [83, 99, 84, 76]}
df = pd.DataFrame(dict)
  
# displaying the DataFrame
df.style
