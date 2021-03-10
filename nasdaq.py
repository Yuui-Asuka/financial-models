from pandas_datareader import data
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas

if not os.path.exists('Nasdaq.csv'):
    Nasdaq = data.DataReader(name='^IXIC', start=2000, data_source='yahoo')
    Nasdaq.to_csv('Nasdaq.csv')
Nasdaq = pandas.read_csv('Nasdaq.csv', index_col='Date', parse_dates=True)
#Nasdaq['close'].plot(alpha=0.5, style='-')
Nasdaq = Nasdaq['Close'].asfreq('D', method='ffill')
rets = np.log(Nasdaq/Nasdaq.shift(1))
rets.plot(alpha=0.5, style='-')
plt.show()

