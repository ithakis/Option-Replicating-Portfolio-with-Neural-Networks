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


class Insurance_Portfolio():
    def __init__(self, parameters, simulate=True):
        self.n_paths_Y = parameters['n_paths_Y']
        self.Y       = parameters['Y']
        self.T       = parameters['T']
        self.mu      = parameters['mu']
        self.r       = parameters['r']
        self.sigma   = parameters['sigma']
        self.dt      = parameters['dt']
        
        self.n_time_steps_Y = ceil(self.T/self.dt)+1

        self.SV     = parameters['SV']
        """ SV parameters """
        self.theta  = self.sigma**2    # Long run variance
        self.kappa  = 0.01         # mean reversion rate
        self.s_vol  = 0.1        # volatility of variance
        # initial variance
        self.kappa_dt    = self.kappa*self.dt
        self.sigma_sdt   = self.s_vol*np.sqrt(self.dt)

        self.reduction   = parameters['reduction']

        self.ADJUSTMENT_FACTOR   = parameters['ADJUSTMENT_FACTOR']
        # numerical percentage to adjust payoff
        self.survival_rate = parameters['survival_rate']

        if simulate:
            self.simualte_Ypaths_Bpaths()
            self.Replicating_Portfolio()
        
        self.simulated = simulate

    def sobol_norm(self, m, d=1 ,seed=1234):
        sampler = qmc.Sobol(d, scramble=True, seed=seed)
        x_sobol = sampler.random_base2(m)
        return stats.norm.ppf(x_sobol)

    def simualte_Ypaths_Bpaths(self, local=False):
        timer=perf_counter()
        """ Simulate Fund Price """
        W1      = self.sobol_norm(self.n_paths_Y, d=self.n_time_steps_Y)
        W_SV    = self.sobol_norm(self.n_paths_Y, d=self.n_time_steps_Y, seed=1235)
        Y_paths = np.empty((2**self.n_paths_Y, self.n_time_steps_Y))
        Y_paths[:,0] = np.log(self.Y)

        """ SV - Verion """
        vt = np.full(shape=(int(2**self.n_paths_Y),self.n_time_steps_Y), fill_value=self.sigma**2) # initial variance 
        print('----------------------------------------------------------------')
        for t in range(1,self.n_time_steps_Y):
            # Y_paths[:,t] = Y_paths[:,t-1] + Y_paths[:,t-1] * (mu*dt + sigma * np.sqrt(dt)*W1[:,t]).squeeze()
            """ Advanced Version : continious time + Stochastic Volatility """
            # Simulate variance processes
            if self.SV:  vt[:,t] = vt[:,t-1] + self.kappa_dt*(self.theta - vt[:,t-1]) + self.sigma_sdt*np.sqrt(vt[:,t-1]*self.dt)*W_SV[:,t]
            volatility = vt[:,t] if self.SV else self.sigma**2
            # Simulate log asset prices
            Y_paths[:,t] = Y_paths[:,t-1] + ((self.mu - 0.5*volatility)*self.dt + np.sqrt(volatility*self.dt)*W1[:,t])

        Y_paths = np.exp(Y_paths)

        """ Generate Bond Data """
        B = np.exp(self.r*np.linspace(0,self.T, self.n_time_steps_Y))
        B = np.broadcast_to(B, Y_paths.shape)

        if not local:
            self.Y_paths         = Y_paths[:, slice(0, None, self.reduction)]
            self.B               = B[:, slice(0, None, self.reduction)]
            self.n_time_steps_Y  = ceil(self.n_time_steps_Y/self.reduction) ; self.dt *= self.reduction

            print(f'Time to simulate: {perf_counter()-timer:.3f}sec')

            self.Payoff_Y       = np.where(Y_paths[:,-1] > self.Y, Y_paths[:,-1], self.Y)
            self.out_of_money_P = np.where(Y_paths[:,-1] < self.Y, 1.0, 0.0).mean()
            self.S              = self.Payoff_Y * self.survival_rate
        else:
            Y_paths         = Y_paths[:, slice(0, None, self.reduction)]
            B               = B[:, slice(0, None, self.reduction)]
            n_time_steps_Y  = ceil(self.n_time_steps_Y/self.reduction) ; self.dt *= self.reduction

            print(f'Time to simulate: {perf_counter()-timer:.3f}sec')

            Payoff_Y       = np.where(Y_paths[:,-1] > self.Y, Y_paths[:,-1], self.Y)
            out_of_money_P = np.where(Y_paths[:,-1] < self.Y, 1.0, 0.0).mean()
            S              = self.Payoff_Y * self.survival_rate

    def Replicating_Portfolio(self):
        def get_phi_psi(model, X0, mean=True):
            layer_output=model.get_layer('Phi_Psi').output
            pw_model = Model(inputs=[model.input], outputs=[layer_output])

            linear_layer_output = pw_model.predict(X0)
            if mean:
                return linear_layer_output[:,0].mean(), linear_layer_output[:,1].mean()
            else:
                return linear_layer_output[:,0], linear_layer_output[:,1]
        
        def scheduler(epoch, lr):
            if epoch < 100 :
                return 1e-2
            elif epoch < 200 :
                return 1e-3
            elif epoch < 400 :
                return 5e-4
            else:
                return lr

        rl_scheduler = keras.callbacks.LearningRateScheduler(scheduler)
        dense_initalizer = keras.initializers.RandomNormal(mean=0, stddev=0.1, seed=1234)
        const_initalizer = keras.initializers.RandomNormal(mean=[1-self.out_of_money_P,self.out_of_money_P], stddev=0.0, seed=1234)

        Input_S = keras.Input(shape=(1,), name='input: S_{t} ') # LeakyReLU
        x = keras.layers.Dense(8, activation='LeakyReLU', name='LeakyReLU_1', kernel_initializer=dense_initalizer)(Input_S)
        x = keras.layers.Dense(8, activation='LeakyReLU', name='LeakyReLU_2', kernel_initializer=dense_initalizer)(x)
        holdings = keras.layers.Dense(2, activation='linear', name='Phi_Psi', kernel_initializer=dense_initalizer, bias_initializer=const_initalizer)(x)

        prices_1 = keras.Input(shape=(2,), name='input: S_{t}, B_{t} ')
        value    = keras.layers.Dot(axes = 1, name='V_t')([holdings, prices_1]) 

        model = keras.Model(inputs=[Input_S, prices_1], outputs=value, name="Replicating_Portfolio")
        # model.summary()
        #------------------------#------------------------#------------------------#------------------------#------------------------#------------------------#------------------------
        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=7, restore_best_weights=True)
        model.compile(optimizer = keras.optimizers.Adam(learning_rate=1e-3),
                    loss = "mse", run_eagerly=False, 
                    metrics=["mae", "mape"])

        values          = np.empty((self.S.shape[0], self.Y_paths.shape[1]))
        values[:,-1]    = self.S

        """ Calculate: Payoff for Options """
        Flag = True ; its = 0
        Errors      = np.zeros((1,2))
        P_E_Values  = np.ones((1,3)) * self.S.mean()
        Phi_Psi_HV  = []
        for t_i in trange(self.n_time_steps_Y-2, -1, -1):
            _Y_t  = self.Y_paths[:,t_i]
            _B_t  = self.B[:,t_i]
            _Y_t1 = self.Y_paths[:,t_i+1]
            _B_t1 = self.B[:,t_i+1]

            X0 = [_Y_t, np.stack((_Y_t, _B_t), axis=-1)]
            X1 = [_Y_t, np.stack((_Y_t1, _B_t1), axis=-1)]

            epochs = 300
            if Flag :
                # print(f'S.mean: {self.S.mean():.5f}\nP.mean: {model.predict(X0, verbose=0, batch_size=512).squeeze().mean():.5f}')
                callabacks = [rl_scheduler, keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)]
                Flag = False 
            else : 
                epochs = 20 
                callabacks = [callback]

            model.fit(X1, values[:,t_i+1], epochs=epochs, validation_split=0.0, verbose=0, batch_size=512, callbacks=callabacks) #, initial_epoch= 200 if t_i != n_time_steps-2 else 0)
            
            """ Get mean phi-psi """
            phi, psi = get_phi_psi(model, X0, mean=False)
            for f  in phi: Phi_Psi_HV.append([f,  t_i*self.dt, 'Phi'])
            for ps in psi: Phi_Psi_HV.append([ps, t_i*self.dt, 'Psi'])

            values[:,t_i] = model.predict(X0, verbose=0, batch_size=512).squeeze()

            self.values = values
            # Errors = np.append(Errors, np.array(model.evaluate(X1, values[:,t_i+1], batch_size=512)[1:]).reshape(1,2), axis=0) ; its += 1
            # P_E_Values = np.append(P_E_Values, np.array([values[:,t_i].mean(), S.mean()*np.exp(-mu*dt*its), S.mean()*np.exp(-r*dt*its)]).reshape(1,3), axis=0)

    def Replicaitng_Portfolio_V0(self, changed_param:tuple = None, return_info=False, make_plots=False):
        if changed_param:   
            temp_stored_original_value = self.parameters[changed_param[1]]
            self.parameters[changed_param[0]] = changed_param[1]

        if not changed_param and not self.simulated:
            self.simualte_Ypaths_Bpaths()
            self.Replicating_Portfolio()
            local_values = self.values
        elif changed_param:
            Ypaths, Bpaths = self.simualte_Ypaths_Bpaths(local=True)
            local_values   = self.Replicating_Portfolio(Ypaths, Bpaths, local=True)
        else:
            local_values = self.values

        if make_plots:
            V0_RP = local_values[:,0].mean()

            qs = np.quantile(local_values, q=[.99, .95, .9, .1, .05, .01], axis=0) 
            t  = np.linspace(0, self.T, self.n_time_steps_Y)

            fig, ax = plt.subplots()
            ax.fill_between(t, qs[0,:].squeeze(), qs[-1,:].squeeze(), alpha=.3)
            ax.fill_between(t, qs[1,:].squeeze(), qs[-2,:].squeeze(), alpha=.5)
            ax.fill_between(t, qs[2,:].squeeze(), qs[-3,:].squeeze(), alpha=.99)
            plt.ticklabel_format(style='plain', useLocale=True)
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

            ax.axhline(V0_RP, color='tab:orange')
            ax.axhline(self.S.mean() * np.exp(-self.mu*self.T) * self.ADJUSTMENT_FACTOR, color='tab:blue')
            ax.axhline(self.S.mean() * np.exp(-self.r*self.T) * self.ADJUSTMENT_FACTOR, color='tab:green')
            ax.legend([r'PV $a=99$',r'PV $a=95$',r'PV $a=90$', r'$V_0$ Replicating Portfolio', r'$V_0$ Discounted $E^P$ Payoff', r'$V_0$ Discounted $E^Q$ Payoff'])
            ax.set_xlabel('T - time') ; ax.set_ylabel('Value of Portfolio') ; ax.set_title('Value of the portfolio over time')

        if changed_param:   
            self.parameters[changed_param[0]] = temp_stored_original_value
        return local_values[:,0].mean(), 




