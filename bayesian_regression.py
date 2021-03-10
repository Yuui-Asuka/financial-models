import os
import numpy as np
import pymc3 as pm
from matplotlib.pylab import date2num, DayLocator, DateFormatter
from pymc3.distributions.timeseries import GaussianRandomWalk
from pandas_datareader import data
import matplotlib.pyplot as plt
import scipy.optimize as sco
import pandas as pd
pd.set_option('display.max_columns', None)


class Bayesian(object):

    def __init__(self):
        self.subsample_alpha = 60
        self.subsample_beta = 60

    def load_data(self):

        if not os.path.exists('gold.csv'):
            item = data.DataReader(name=['GLD', 'SLV'], start=2007, data_source='yahoo')['Close']
            item.to_csv('gold.csv')
        else:
            item = pd.read_csv('gold.csv', index_col='Date', parse_dates=True)
            self.item = item.dropna()
        print(item.corr())
        item.plot()

    def plot_figure(self):
        item = self.item
        mpl_dates = date2num(item.index)
        plt.figure(figsize=(8, 4))
        plt.scatter(item['GLD'], item['SLV'], c=mpl_dates, marker='o', s=8)
        plt.grid(True)
        plt.xlabel('GLD')
        plt.ylabel('SLV')
        plt.colorbar(ticks=DayLocator(interval=250), format=DateFormatter('%d %b %y'))

    def regression(self):
        item = self.item
        mpl_dates = date2num(item.index)
        with pm.Model() as model:
            alpha = pm.Normal(name='alpha', mu=0, sd=20)
            beta = pm.Normal(name='beta', mu=0, sd=20)
            sigma = pm.Uniform(name='sigma', lower=0, upper=50)
            y_est = alpha + beta * item['SLV'].values
            likelihood = pm.Normal(name='GLD', mu=y_est, sd=sigma, observed=item['GLD'].values)
            start = pm.find_MAP()
            step = pm.NUTS(scaling=start)
            trace = pm.sample(500, step=step, start=start, progressbar=False, chains=1, tune=2000)

        fig = pm.traceplot(trace)
        plt.figure(figsize=(8, 4))
        plt.scatter(item['SLV'], item['GLD'], c=mpl_dates, marker='o')
        plt.grid(True)
        plt.xlabel('SLV')
        plt.ylabel('GLD')
        for i in range(len(trace)):
            plt.plot(item['SLV'], trace['alpha'][i] + trace['beta'][i] * item['SLV'])
        plt.colorbar(ticks=DayLocator(interval=250), format=DateFormatter('%d %b %y'))

    def random_regression(self):
        model_randomwalk = pm.Model()
        item = self.item[:3180]
        mpl_dates = date2num(item.index)
        with model_randomwalk:
            sigma_alpha = pm.Exponential('sigam_alpha', 1.0 / 0.02, testval=0.1)
            sigma_beta = pm.Exponential('sigma_beta', 1.0 / 0.02, testval=0.1)

            alpha = GaussianRandomWalk('alpha', sigma_alpha ** -2, shape=int(len(item) / self.subsample_alpha))
            beta = GaussianRandomWalk('beta', sigma_beta ** -2, shape=int(len(item) / self.subsample_beta))
            alpha_r = np.repeat(alpha, self.subsample_alpha)
            beta_r = np.repeat(beta, self.subsample_beta)

            regression = alpha_r + beta_r * item.SLV.values
            sd = pm.Uniform(name='sd', lower=0, upper=20)
            likelihood = pm.Normal(name='GLD', mu=regression, sd=sd,
                           observed=item.GLD.values)

            start = pm.find_MAP(vars=[alpha, beta], fmin=sco.fmin_l_bfgs_b)
            step = pm.NUTS(scaling=start)
            trace_rw = pm.sample(500, step, start=start, progressbar=False, tune=2000)

        part_dates = np.linspace(min(mpl_dates), max(mpl_dates), 53)

        fig, ax1 = plt.subplots(figsize=(10, 5))
        plt.plot(part_dates, np.mean(trace_rw['alpha'], axis=0), 'b', lw=2.5, label='alpha')
        for i in range(10, 55):
            plt.plot(part_dates, trace_rw['alpha'][i], 'b-.', lw=0.75)
        plt.xlabel('date')
        plt.ylabel('alpha')
        plt.axis('tight')
        plt.grid(True)
        plt.legend(loc=2)
        ax1.xaxis.set_major_formatter(DateFormatter('%d %b %y'))
        ax2 = ax1.twinx()
        plt.plot(part_dates, np.mean(trace_rw['beta'], axis=0), 'r', lw=2.5, label='beta')
        for i in range(10, 55):
            plt.plot(part_dates, trace_rw['beta'][i], 'r-.', lw=0.75)
        plt.ylabel('beta')
        plt.legend(loc=4)
        fig.autofmt_xdate()

        plt.figure(figsize=(10, 5))
        plt.scatter(item['SLV'], item['GLD'], c=mpl_dates[:3180], marker='o')
        plt.colorbar(ticks=DayLocator(interval=250), format=DateFormatter('%d %b %y'))
        plt.grid(True)
        plt.xlabel('SLV')
        plt.ylabel('GLD')
        x = np.linspace(min(item['SLV']), max(item['SLV']))
        for i in range(53):
            alpha_rw = np.mean(trace_rw['alpha'].T[i])
            beta_rw = np.mean(trace_rw['beta'].T[i])
            plt.plot(x, alpha_rw + beta_rw * x, color=plt.cm.jet(256 * i / 53))

    def show_figure(self):
        self.load_data()
        self.plot_figure()
        self.regression()
        self.random_regression()
        plt.show()

BYS = Bayesian()
BYS.show_figure()

