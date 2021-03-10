import warnings
import os
import pandas as pd
import pymc3 as pm
import numpy as np
from matplotlib.pylab import date2num, DayLocator, DateFormatter
from pandas_datareader import data
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
warnings.simplefilter('ignore')

symbols = ['ATVI', 'ADBE', 'AKAM', 'ALTR', 'AMZN', 'AMGN', 'AAPL', 'ADSK',
           'BBBY', 'BIIB', 'INTC', 'CHRW', 'LEAP', 'CELG', 'FMCN', 'CHKP',
           'CTAS', 'CSCO', 'COST', 'DELL', 'DISH', 'EBAY', 'EXPD', 'FAST',
           'FLIR', 'GRMN', 'GOOG', 'HSIC', 'INFY', 'INTU', 'ISRG', 'KLAC',
           'QCOM', 'LOGI', 'TEVA', 'MNST', 'NTAP', 'NVDA', 'ORCL', 'PDCO',
           'WYNN', 'XLNX', 'SIRI', 'CMCSA', 'MSFT', 'MCHP', 'VRSN', 'VMED',
           '^IXIC']
if not os.path.exists('nas.csv'):
    stocks = pd.DataFrame()
    for sym in symbols:
        stocks[sym] = data.DataReader(sym, data_source='yahoo', start=2008)['Close']
    stocks.to_csv('nas.csv')
price = pd.read_csv('nas.csv', index_col='Date', parse_dates=True)
price = price.dropna(axis='columns')
Nasdaq = pd.DataFrame(price.pop('^IXIC'))

def scale_function(x):
    return (x - x.mean()) / x.std()

def get_we(x):
    return x / x.sum()

pca = KernelPCA(n_components=8).fit(price.apply(scale_function))
pca_components = pca.transform(-price)
weights = get_we(pca.lambdas_)
Nasdaq['PCA_8'] = np.dot(pca_components, weights)
mpl_dates = date2num(price.index)
plt.figure(figsize=(8, 4))
plt.scatter(Nasdaq['PCA_8'], Nasdaq['^IXIC'], c=mpl_dates)
lin_reg = np.polyval(np.polyfit(Nasdaq['PCA_8'], Nasdaq['^IXIC'], 1), Nasdaq['PCA_8'])
plt.plot(Nasdaq['PCA_8'], lin_reg, 'r', lw=3)
plt.grid(True)
plt.xlabel('PCA_8')
plt.ylabel('^IXIC')
plt.colorbar(ticks=DayLocator(interval=250), format=DateFormatter('%d %b %y'))
plt.show()



print(len(pca.lambdas_))








