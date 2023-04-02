import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
# from tensorflow_addons.losses import pinball_loss
from tensorflow import function as tf_fun
from tensorflow import keras
from tensorflow.keras import Model
from math import log, sqrt, exp, erf, ceil, floor
from tqdm.notebook import tqdm, trange
import pandas as pd
from math import ceil

import scipy.stats as stats
from scipy.stats import qmc
import matplotlib.pyplot as plt
import seaborn as sns

from time import perf_counter
from scipy.stats import norm
from matplotlib import ticker

from importlib import reload

from numpy.random import seed   
from tensorflow.random import set_seed as tf_seed

tf_seed(1234) ; seed(1234)

def Replicating_Portfolio(**params):
    """ Financial parameters """
    Y       = params['Y']
    K       = params['K']
    T       = params['T']
    mu      = params['mu']
    r       = params['r']
    sigma   = params['sigma']
    rebalancing = params['rebalancing']

    N   = params['N']
    P   = params['P']
    x   = params['x']
    l0  = params['l0']
    c   = params['c']
    ita = params['ita']

    dt      = params['dt']
    n_paths  = params['n_paths']

    n_time_steps = ceil(T/dt)+1


    def sobol_norm(m, d=1 ,seed=1234):
        sampler = qmc.Sobol(d, scramble=True, seed=seed)
        x_sobol = sampler.random_base2(m)
        return stats.norm.ppf(x_sobol)
    
    """ Generate Y paths """
    W1      = sobol_norm(n_paths, d=n_time_steps, seed=1235)
    Y_paths = np.empty((2**n_paths, n_time_steps))
    Y_paths[:,0] = Y

    for t in range(1,n_time_steps):
        Y_paths[:,t] = Y_paths[:,t-1] + Y_paths[:,t-1] * (mu*dt + sigma * np.sqrt(dt)*W1[:,t]).squeeze()

    """ Generate B paths """
    B = np.exp(r*np.linspace(0,T, n_time_steps))
    B = np.broadcast_to(B, Y_paths.shape)

    """ Generate N paths"""
    W2      = sobol_norm(n_paths, d=n_time_steps)
    L_paths = np.empty((2**n_paths, n_time_steps))
    L_paths[:,0] = l0
    for t in range(1,n_time_steps):
        L_paths[:,t] = L_paths[:,t-1] + (c * L_paths[:,t-1] * dt + ita * np.sqrt(dt)*W2[:,t]).squeeze()

    N_paths = np.empty((2**n_paths, n_time_steps), dtype=int)
    N_paths[:,0] = N
    # Binomial Distribution of N(t)
    for t in range(1,n_time_steps):
        probabilities = np.exp(-L_paths[:,t]*dt)
        np.random.seed(1234+t) # to ensure reproducible results
        N_paths[:,t]  = np.random.binomial(N_paths[:,t-1], probabilities)

    """ Summary Statistics """
    E_N_T   = N_paths[:,-1].mean()
    Payoff_Y     = np.where(Y_paths[:,-1] > Y, Y_paths[:,-1], Y)
    S_T = Payoff_Y * P * N_paths[:,-1]
    out_of_money_P  = np.where(Y_paths[:,-1] < Y, 1.0, 0.0).mean()

    """ """
    reduction       = floor((n_time_steps-1)/(T/rebalancing))   ; print(f'reduction = {reduction}')
    Y_paths         = Y_paths[:, slice(0, None, reduction)]
    B               = B[:, slice(0, None, reduction)]
    N_paths         = N_paths[:, slice(0, None, reduction)]
    n_time_steps  = ceil(n_time_steps/reduction) ; dt = dt *(reduction)
    Y_paths[:,-1].mean(), N_paths[:,-1].mean(), dt, Y_paths.shape, n_time_steps

    """ Liability S(T) """
    S_T = Payoff_Y * N_paths[:, -1] * P
    _in = {'S_T': S_T, 'Y_T': Y_paths[:,-1]}
    ax = pd.DataFrame.from_dict(_in).plot(kind='scatter', x='Y_T', y='S_T',xlabel=r'$Y_T$',ylabel=r'$S_T$ - EUR', title=r'Payoff $S_T$ Relative to $Y_T$', marker='.', color='tab:blue')
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    print(f'Expected S_T = {S_T.mean():,.0f} EUR')

