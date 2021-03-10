import warnings
import pandas as pd
import pymc3 as pm
import numpy as np
from pandas_datareader import data
import matplotlib.pyplot as plt
warnings.simplefilter('ignore')
item = data.DataReader(name='GLD', start=2000, data_source='yahoo', pause=0.001)['Close']
#nas = pd.read_csv('Nasdaq.csv', index_col='Date')['Close'].dropna()
item = item.dropna().reset_index(drop=True)
item.to_csv('close.csv')
