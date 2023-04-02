import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from math import ceil
import numpy as np
from time import perf_counter
import scipy.stats as stats
from scipy.stats import qmc
from tensorflow import keras
from tensorflow.keras import Model

import matplotlib.pyplot as plt
from matplotlib import ticker

from tqdm import trange

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
    x   = params['x']
    l0  = params['l0']
    c   = params['c']
    ita = params['ita']

    dt      = params['dt']
    n_paths  = params['n_paths']

    n_time_steps = ceil(T/dt)+1
    """ Simulate Fund Price """
    def sobol_norm(m, d=1 ,seed=1234):
        sampler = qmc.Sobol(d, scramble=True, seed=seed)
        x_sobol = sampler.random_base2(m)
        return stats.norm.ppf(x_sobol)
    
    W1      = sobol_norm(n_paths, d=n_time_steps, seed=1235)
    Y_paths = np.empty((2**n_paths, n_time_steps))
    Y_paths[:,0] = Y

    print('----------------------------------------------------------------')
    for t in range(1,n_time_steps):
        Y_paths[:,t] = Y_paths[:,t-1] + Y_paths[:,t-1] * (mu*dt + sigma * np.sqrt(dt)*W1[:,t]).squeeze()

    """ Generate Bond Data """
    B = np.exp(r*np.linspace(0,T, n_time_steps))
    B = np.broadcast_to(B, Y_paths.shape)