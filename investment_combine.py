from pandas_datareader import data
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import scipy.optimize as sco
import scipy.interpolate as sci
plt.style.use('ggplot')

stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'GE']
if not os.path.exists('AAPL.csv'):
    for s in stocks:
        price = data.DataReader(name=s, start=2005, data_source='yahoo')
        price.to_csv('{}.csv'.format(s))
kw = []
for s in stocks:
    kwargs = dict(filepath_or_buffer='{}.csv'.format(s), index_col='Date', parse_dates=True)
    kw.append(kwargs)
apple = pd.read_csv(**kw[0])
microsoft = pd.read_csv(**kw[1])
google = pd.read_csv(**kw[2])
amazon = pd.read_csv(**kw[3])
general = pd.read_csv(**kw[4])
stocks_data = pd.DataFrame()
for s, n in zip([apple, microsoft, google, amazon, general], stocks):
     stocks_data[n] = s['Close']
stocks_data.asfreq('D', method='ffill')
nor = (stocks_data / stocks_data.ix[0]*100)
pay = np.log(nor / nor.shift(1))
avg = np.full(shape=5, fill_value=1. / 5)


def statistics(weights):
    weights = np.array(weights)
    weights /= np.sum(weights)
    p_ret = np.sum(pay.mean() * weights) * 252
    p_vol = np.sqrt(np.dot(weights.T, np.dot(pay.cov() * 252, weights)))
    return np.array([p_ret, p_vol, p_ret / p_vol])


def min_func_sharpe(weights):
    return -statistics(weights)[2]


def min_func_variance(weights):
    return statistics(weights)[1] ** 2


def min_func_port(weights):
    return statistics(weights)[1]


def caculate_opt():
    cons = ({'type': 'eq',  'fun': lambda x: np.sum(x) - 1})
    bnds = tuple((0, 1) for x in range(5))
    avg = np.full(shape=5, fill_value=1./5)
    opts = sco.minimize(min_func_sharpe, avg, method='SLSQP', bounds=bnds, constraints=cons)
    optv = sco.minimize(min_func_variance, avg, method='SLSQP', bounds=bnds, constraints=cons)
    return opts, optv


def monte_carlo_simulation():
    prets = []
    pvols = []
    for p in range(2500):
        weights = np.random.random(5)
        weights /= np.sum(weights)
        prets.append(np.sum(pay.mean() * weights) * 252)
        pvols.append(np.sqrt(np.dot(weights.T, np.dot(pay.cov() * 252, weights))))
    prets = np.array(prets)
    pvols = np.array(pvols)
    plt.figure(figsize=(8, 4))
    plt.scatter(pvols, prets, c=prets / pvols, marker='o')
    plt.grid(True)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe ratio')
    plt.show()
    return prets, pvols


def effect_edge(opts, optv, prets, pvols):
    trets = np.linspace(0.0, 0.25, 50)
    tvols = []
    for tret in trets:
        weights = np.random.random(5)
        weights /= np.sum(weights)
        cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - tret},
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0, 1) for x in weights)
        res = sco.minimize(min_func_port, avg, method='SLSQP', bounds=bnds, constraints=cons)
        tvols.append(res['fun'])
    tvols = np.array(tvols)
    plt.figure(figsize=(8, 4))
    plt.scatter(pvols, prets, c=prets / pvols, marker='o')
    plt.scatter(tvols, trets, c=trets / tvols, marker='x')
    plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0],
             'r*', markersize=15.0)
    plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0],
             'y*', markersize=15.0)
    plt.grid(True)
    plt.xlabel('expected volatility')
    plt.ylabel('ecpected return')
    plt.colorbar(label='Sharpe ratio')
    plt.show()
    return trets, tvols


opts, optv = caculate_opt()
prets, pvols = monte_carlo_simulation()
trets, tvols = effect_edge(opts, optv, prets, pvols)

ind = np.argmin(tvols)
evols = tvols[ind:]
erets = trets[ind:]
tck = sci.splrep(evols, erets)


def f(x):
    return sci.splev(x, tck, der=0)


def df(x):
    return sci.splev(x, tck, der=1)


def equations(p, rf=0.01):
    eq1 = rf - p[0]
    eq2 = rf + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3


opt = sco.fsolve(equations, np.array([0.01, 0.5, 0.15]))
np.round(equations(opt), 6)
plt.figure()
plt.scatter(pvols, prets, c=(prets - 0.01) / pvols, marker='o')
plt.plot(evols, erets, 'g', lw=4.0)
cx = np.linspace(0.0, 0.3)
plt.plot(cx, opt[0] + opt[1] * cx, lw=1.5)
plt.plot(opt[2], f(opt[2]), 'r*', markersize=15.0)
plt.grid(True)
plt.axhline(0, color='k', ls='--', lw=2.0)
plt.axvline(0, color='k', ls='--', lw=2.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()
cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - f(opt[2])},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
weights = np.random.random(5)
weights /= np.sum(weights)
bnds = tuple((0, 1) for x in weights)
res = sco.minimize(min_func_port, avg, method='SLSQP', bounds=bnds, constraints=cons)
print(res['x'].round(3))





