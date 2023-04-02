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

def Replicating_Portfolio(params):
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

    ADJUSTMENT_FACTOR = N * P

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
    Payoff_Y     = np.where(Y_paths[:,-1] > K, Y_paths[:,-1], K)
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

    dense_initalizer = keras.initializers.RandomNormal(mean=0, stddev=0.1, seed=1234)
    const_initalizer = keras.initializers.RandomNormal(mean=[1-out_of_money_P,out_of_money_P], stddev=0.0, seed=1234)

    # model 1
    Input_S_N = keras.Input(shape=(3,), name='input: S_{t}, N_{t}, l_{t}') 
    x = keras.layers.Dense(8, activation='LeakyReLU', name='LeakyReLU_1', kernel_initializer=dense_initalizer)(Input_S_N)
    x = keras.layers.Dense(8, activation='LeakyReLU', name='LeakyReLU_2', kernel_initializer=dense_initalizer)(x)
    holdings = keras.layers.Dense(2, activation='linear', name='Phi_Psi', kernel_initializer=dense_initalizer, bias_initializer=const_initalizer)(x)

    prices_1 = keras.Input(shape=(2,), name='input: S_{t}, B_{t}')
    S_out    = keras.layers.Dot(axes = 1, name='V_t')([holdings, prices_1]) 

    model1 = keras.Model(inputs=[Input_S_N, prices_1], outputs=S_out, name="Replicating_Portfolio_MSE")

    #model2
    Input_S_N2 = keras.Input(shape=(3,), name='input: S_{t}, N_{t}, l_{t}') 
    x2 = keras.layers.Dense(8, activation='LeakyReLU', name='LeakyReLU_1', kernel_initializer=dense_initalizer)(Input_S_N2)
    x2 = keras.layers.Dense(8, activation='LeakyReLU', name='LeakyReLU_2', kernel_initializer=dense_initalizer)(x2)
    holdings2 = keras.layers.Dense(2, activation='linear', name='Phi_Psi', kernel_initializer=dense_initalizer, bias_initializer=const_initalizer)(x2)

    prices_12 = keras.Input(shape=(2,), name='input: S_{t}, B_{t}')
    S_out2    = keras.layers.Dot(axes = 1, name='V_t')([holdings2, prices_12]) 

    model2 = keras.Model(inputs=[Input_S_N, prices_1], outputs=S_out, name="Replicating_Portfolio_Q99")

    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=7, restore_best_weights=True)
    model1.compile(optimizer = keras.optimizers.Adam(learning_rate=1e-3),
                loss = 'mse', run_eagerly=False, 
                metrics=["mae", "mape"])
    model2.compile(optimizer = keras.optimizers.Adam(learning_rate=1e-3),
                loss = quantile_loss_f, run_eagerly=False, 
                metrics=["mae", "mape"])

    N_paths_NN      = N_paths / N
    values          = np.empty_like(Y_paths)
    values[:,-1]    = Payoff_Y * N_paths_NN[:,-1] 
    E_payoff        = (values[:,-1]).mean()

    """ Calculate: Payoff for Options """
    Flag = True ; its = 0
    Errors      = np.zeros((1,2))
    P_E_Values  = np.ones((1,3)) * values[:,-1].mean()
    VaR_HV      = []
    Phi_Psi_HV  = []
    for t_i in trange(n_time_steps-2, -1, -1):
        print(f'Y_({(t_i+1)*dt:.2f}) = {Y_paths[:,t_i+1].mean():.3f}, N_({(t_i+1)*dt:.2f}) = {N_paths_NN[:,t_i+1].mean():.3f}')
        _Y_t  = Y_paths[:,t_i]
        _B_t  = B[:,t_i]
        _Y_t1 = Y_paths[:,t_i+1]
        _B_t1 = B[:,t_i+1]

        X0 = [np.stack((_Y_t, N_paths_NN[:,t_i], L_paths[:,t_i]), axis=-1), np.stack((_Y_t, _B_t), axis=-1)]
        X1 = [np.stack((_Y_t, N_paths_NN[:,t_i], L_paths[:,t_i]), axis=-1), np.stack((_Y_t1, _B_t1), axis=-1)]

        epochs = 500
        if Flag :
            # print(f'S.mean: {S.mean():.5f}\nP.mean: {model.predict(X0, verbose=0, batch_size=512).squeeze().mean():.5f}')
            callabacks = [lr_scheduler, keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)]
        else : 
            epochs = 100 
            callabacks = [callback]

        model1.fit(X1, values[:,t_i+1], epochs=epochs, validation_split=0.0, verbose=0, batch_size=512, callbacks=callabacks) #, initial_epoch= 200 if t_i != n_time_steps-2 else 0)
        g_t  = model1.predict(X0, verbose=0, batch_size=512).squeeze()
        g_t1 = model1.predict(X1, verbose=0, batch_size=512).squeeze()

        Errors = np.append(Errors, np.array(model1.evaluate(X1, values[:,t_i+1], batch_size=512)[1:]).reshape(1,2), axis=0)
        
        model2.fit(X1, values[:,t_i+1], epochs=epochs, validation_split=0.0, verbose=0, batch_size=512, callbacks=callabacks) #, initial_epoch= 200 if t_i != n_time_steps-2 else 0)    
        h_t  = model2.predict(X0, verbose=0, batch_size=512).squeeze()
        h_t1 = model2.predict(X1, verbose=0, batch_size=512).squeeze()

        values[:,t_i] = g_t + cost_of_capital*(h_t - g_t)

        """ Update phi-psi and VaR """
        Phi_Psi_HV, VaR_HV = get_phi_psi_VaR(model1, model2, X0, Phi_Psi_HV, VaR_HV, cost_of_capital, values[:,t_i+1], _Y_t1, _B_t1)

        its += 1 ; Flag = False 
        P_E_Values = np.append(P_E_Values, np.array([values[:,t_i].mean(), E_payoff*np.exp(-mu*dt*its), E_payoff*np.exp(-r*dt*its)]).reshape(1,3), axis=0)

        phi_psi_df = pd.DataFrame(Phi_Psi_HV, columns=['Value', 'T', 'Type'])
        phi_psi_df.Value *= ADJUSTMENT_FACTOR 
        _ppg = phi_psi_df.groupby(['T', 'Type']).agg('mean')

        phi = _ppg.loc[(0, "Phi")].mean()
        psi = _ppg.loc[(0, "Psi")].mean()
        return phi