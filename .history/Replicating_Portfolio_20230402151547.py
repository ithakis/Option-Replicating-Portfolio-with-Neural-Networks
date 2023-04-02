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

    """ Neural Network Hedging """
    def get_phi_psi_VaR(model1, model2, X0, Phi_Psi_HV, VaR_HV, cost_of_capital, values, Y_t, B_t):
        layer_output1=model1.get_layer('Phi_Psi').output
        pw_model1 = Model(inputs=[model1.input], outputs=[layer_output1])

        linear_layer_output1 = pw_model1.predict(X0)

        layer_output2 = model2.get_layer('Phi_Psi').output
        pw_model2 = Model(inputs=[model2.input], outputs=[layer_output2])

        linear_layer_output2 = pw_model2.predict(X0)

        phi = linear_layer_output1[:,0] + cost_of_capital * (linear_layer_output1[:,0] - linear_layer_output2[:,0])
        psi = linear_layer_output1[:,1] + cost_of_capital * (linear_layer_output1[:,1] - linear_layer_output2[:,1])

        for f  in phi: Phi_Psi_HV.append([f,  t_i*dt, 'Phi'])
        for ps in psi: Phi_Psi_HV.append([ps, t_i*dt, 'Psi'])

        VaR = values - phi * Y_t - psi * B_t

        print(f'VaR: {np.quantile(VaR, .98):4f} (98%),  {np.quantile(VaR, .99):4f} (99%)')
        for var in VaR: VaR_HV.append([var, (t_i+1)*dt])
        
        return Phi_Psi_HV, VaR_HV


    def scheduler(epoch, lr):
        if epoch < 100 :
            return 1e-2
        elif epoch < 200 :
            return 1e-3
        elif epoch < 400 :
            return 5e-4
        else:
            return lr

    @tf_fun
    def quantile_loss(y, y_p):
            QUANTILE = .99
            e = y-y_p
            return keras.backend.mean(keras.backend.maximum(QUANTILE*e, (QUANTILE-1)*e))

    lr_scheduler    = keras.callbacks.LearningRateScheduler(scheduler)
    quantile_loss_f = lambda y, y_p : quantile_loss(y, y_p)

    cost_of_capital = .1 
