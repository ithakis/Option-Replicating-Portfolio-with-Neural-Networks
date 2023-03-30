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
    def __init__(self, parameters, simulate=False):
        self.parameters = parameters
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
            self.Replicating_Portfolio(Y_paths=self.Y_paths, B=self.B, n_time_steps_Y=self.n_time_steps_Y, out_of_money_P=self.out_of_money_P, S=self.S)
        
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
        # print('----------------------------------------------------------------')
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

            # print(f'Time to simulate: {perf_counter()-timer:.3f}sec')

            self.Payoff_Y       = np.where(Y_paths[:,-1] > self.Y, Y_paths[:,-1], self.Y)
            self.out_of_money_P = np.where(Y_paths[:,-1] < self.Y, 1.0, 0.0).mean()
            self.S              = self.Payoff_Y * self.survival_rate
        else:
            Y_paths         = Y_paths[:, slice(0, None, self.reduction)]
            B               = B[:, slice(0, None, self.reduction)]
            n_time_steps_Y  = ceil(self.n_time_steps_Y/self.reduction) ; self.dt *= self.reduction

            # print(f'Time to simulate: {perf_counter()-timer:.3f}sec')

            Payoff_Y       = np.where(Y_paths[:,-1] > self.Y, Y_paths[:,-1], self.Y)
            out_of_money_P = np.where(Y_paths[:,-1] < self.Y, 1.0, 0.0).mean()
            S              = Payoff_Y * self.survival_rate
            return Y_paths, B, n_time_steps_Y, Payoff_Y, out_of_money_P, S


    def Replicating_Portfolio(self, Y_paths, B, n_time_steps_Y, out_of_money_P, S, local=False):
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
        const_initalizer = keras.initializers.RandomNormal(mean=[1-out_of_money_P,out_of_money_P], stddev=0.0, seed=1234)

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

        values          = np.empty((S.shape[0], Y_paths.shape[1]))
        values[:,-1]    = S

        """ Calculate: Payoff for Options """
        Flag = True ; its = 0
        Errors      = np.zeros((1,2))
        P_E_Values  = np.ones((1,3)) * S.mean()
        Phi_Psi_HV  = []
        for t_i in trange(n_time_steps_Y-2, -1, -1):
            _Y_t  = Y_paths[:,t_i]
            _B_t  = B[:,t_i]
            _Y_t1 = Y_paths[:,t_i+1]
            _B_t1 = B[:,t_i+1]

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

            if not local:
                self.values = values
                self.Phi_Psi_HV = (phi[0], psi[0])
            else: 
                return values, (phi[0], psi[0])
            # Errors = np.append(Errors, np.array(model.evaluate(X1, values[:,t_i+1], batch_size=512)[1:]).reshape(1,2), axis=0) ; its += 1
            # P_E_Values = np.append(P_E_Values, np.array([values[:,t_i].mean(), S.mean()*np.exp(-mu*dt*its), S.mean()*np.exp(-r*dt*its)]).reshape(1,3), axis=0)

    def Replicaitng_Portfolio_V0(self, changed_param:dict = None, print_info=False, make_plots=False, return_phi_psi=False):

        if changed_param:
            temp_stored_original_value = []
            for key in changed_param:
                temp_stored_original_value.append([key, self.parameters[key]])
                self.parameters[key] = changed_param[key]
            self.__init__(self.parameters)

        if changed_param and not self.simulated:
            self.simualte_Ypaths_Bpaths()
            self.Replicating_Portfolio(Y_paths=self.Y_paths, B=self.B, n_time_steps_Y=self.n_time_steps_Y, out_of_money_P=self.out_of_money_P, S=self.S )
            # Local Values
            Y_paths=self.Y_paths; B=self.B; n_time_steps_Y=self.n_time_steps_Y; out_of_money_P=self.out_of_money_P; S=self.S
            local_values = self.values
        elif changed_param:   
            Y_paths, B, n_time_steps_Y, Payoff_Y, out_of_money_P, S = self.simualte_Ypaths_Bpaths(local=True)
            if not return_phi_psi:
                local_values   = self.Replicating_Portfolio(Y_paths=Y_paths, B=B, n_time_steps_Y=n_time_steps_Y, out_of_money_P=out_of_money_P, S=S, local=True)
            if return_phi_psi:
                local_values, Phi_Psi_HV   = self.Replicating_Portfolio(Y_paths=Y_paths, B=B, n_time_steps_Y=n_time_steps_Y, out_of_money_P=out_of_money_P, S=S, local=True)
        else:
            # Local Values
            Y_paths=self.Y_paths; B=self.B; n_time_steps_Y=self.n_time_steps_Y; out_of_money_P=self.out_of_money_P; S=self.S
            local_values = self.values

        if make_plots:
            V0_RP = local_values[:,0].mean() * self.ADJUSTMENT_FACTOR

            qs = np.quantile(local_values, q=[.99, .95, .9, .1, .05, .01], axis=0) * self.ADJUSTMENT_FACTOR
            t  = np.linspace(0, self.T, self.n_time_steps_Y)

            fig, ax = plt.subplots()
            ax.fill_between(t, qs[0,:].squeeze(), qs[-1,:].squeeze(), alpha=.3)
            ax.fill_between(t, qs[1,:].squeeze(), qs[-2,:].squeeze(), alpha=.5)
            ax.fill_between(t, qs[2,:].squeeze(), qs[-3,:].squeeze(), alpha=.99)
            plt.ticklabel_format(style='plain', useLocale=True)
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

            ax.axhline(V0_RP, color='tab:orange')
            ax.axhline(S.mean() * np.exp(-self.mu*self.T) * self.ADJUSTMENT_FACTOR, color='tab:blue')
            ax.axhline(S.mean() * np.exp(-self.r*self.T) * self.ADJUSTMENT_FACTOR, color='tab:green')
            ax.legend([r'PV $a=99$',r'PV $a=95$',r'PV $a=90$', r'$V_0$ Replicating Portfolio', r'$V_0$ Discounted $E^P$ Payoff', r'$V_0$ Discounted $E^Q$ Payoff'])
            ax.set_xlabel('T - time') ; ax.set_ylabel('Value of Portfolio') ; ax.set_title('Value of the portfolio over time')

        if print_info:
            print('ADJUSTED \n--------------------------------------------')
            print(f'Value at t_0 (Replicating-P)  = {V0_RP:,.3f}')
            print(f'Discounted Eq[S] ADJ          = {S.mean() * np.exp(-self.r*self.T) * self.ADJUSTMENT_FACTOR:,.3f}')
            print(f' Difference                   = {V0_RP - S.mean() * np.exp(-self.r*self.T) * self.ADJUSTMENT_FACTOR:,.3f} : {(V0_RP - S.mean() * np.exp(-self.r*self.T) * self.ADJUSTMENT_FACTOR)/V0_RP * 100:,.3f}% ')
            print(f'Discounted Ep[S] ADJ          = {S.mean() * np.exp(-self.mu*self.T) * self.ADJUSTMENT_FACTOR:,.3f}')
            print(f' Difference                   = {V0_RP - S.mean() * np.exp(-self.mu*self.T) * self.ADJUSTMENT_FACTOR:,.3f}   : {(V0_RP - S.mean() * np.exp(-self.mu*self.T) * self.ADJUSTMENT_FACTOR)/V0_RP * 100:,.3f}% ')
            print(f'Total Premium at t0           = {self.ADJUSTMENT_FACTOR:,} \nProfit w/o TC                 = {self.ADJUSTMENT_FACTOR - V0_RP:,.0f}')

        if changed_param:   
            for item in temp_stored_original_value:
                self.parameters[item[0]] = item[1]
            self.__init__(self.parameters, simulate=False)
        
        if return_phi_psi:
            return local_values[:,0].mean() * self.ADJUSTMENT_FACTOR, self.Phi_Psi_HV

    
def Simulate_N_T_outcomes(params: dict):
    def sobol_norm(m, d=1 ,seed=1234):
        sampler = qmc.Sobol(d, scramble=True, seed=seed)
        x_sobol = sampler.random_base2(m)
        return stats.norm.ppf(x_sobol)

    n_time_steps = int(params['T']/params['dt'])+1 #; print(f'Number of steps: {n_time_steps}')

    W2      = sobol_norm(params['n_paths_N'], d=n_time_steps)
    L_paths = np.empty((2**params['n_paths_N'], n_time_steps))
    L_paths[:,0] = params['l0']
    for t in range(1,n_time_steps):
        L_paths[:,t] = L_paths[:,t-1] + (params['c'] * L_paths[:,t-1] * params['dt'] + params['ita'] * np.sqrt(params['dt'])*W2[:,t]).squeeze()

    N_paths = np.empty((2**params['n_paths_N'], n_time_steps), dtype=int)
    N_paths[:,0] = params['N']
    """ Binomial Distribution of N(t) """
    for t in range(1,n_time_steps):
        probabilities = np.exp(-L_paths[:,t]*params['dt'])
        np.random.seed(1234+t) # to ensure reproducible results
        N_paths[:,t]  = np.random.binomial(N_paths[:,t-1], probabilities)

    return N_paths[:,-1]

def Calculate_V0_Phi_Psi(**params):
    def sobol_norm(m, d=1 ,seed=1234):
        sampler = qmc.Sobol(d, scramble=True, seed=seed)
        x_sobol = sampler.random_base2(m)
        return stats.norm.ppf(x_sobol)

    ADJUSTMENT_FACTOR = params['N'] * 100
    n_paths_Y   = params['n_paths_Y']

    Y       = params['Y']
    K       = params['K']
    T       = params['T'] # Years
    mu      = params['mu']
    r       = params['r']
    sigma   = params['sigma']
    dt      = params['dt']

    n_time_steps_Y = ceil(T/dt)+1 #; print(f'Number of steps: {n_time_steps_Y} \ndt         = {dt:.3f} year')

    SV = params['SV']
    """ SV parameters """
    theta = sigma**2    # Long run variance
    kappa = params['kappa']         # mean reversion rate
    s_vol = params['s_vol']        # volatility of variance
    # initial variance
    kappa_dt    = kappa*dt
    sigma_sdt   = s_vol*np.sqrt(dt) ; timer=perf_counter()

    """ Simulate Fund Price """
    W1      = sobol_norm(n_paths_Y, d=n_time_steps_Y)
    W_SV    = sobol_norm(n_paths_Y, d=n_time_steps_Y, seed=1235)
    Y_paths = np.empty((2**n_paths_Y, n_time_steps_Y))
    Y_paths[:,0] = np.log(Y)

    """ SV - Verion """
    vt = np.full(shape=(int(2**n_paths_Y),n_time_steps_Y), fill_value=sigma**2) # initial variance 
    for t in range(1,n_time_steps_Y):
        # Y_paths[:,t] = Y_paths[:,t-1] + Y_paths[:,t-1] * (mu*dt + sigma * np.sqrt(dt)*W1[:,t]).squeeze()
        """ Advanced Version : continious time + Stochastic Volatility """
        # Simulate variance processes
        if SV:  vt[:,t] = vt[:,t-1] + kappa_dt*(theta - vt[:,t-1]) + sigma_sdt*np.sqrt(vt[:,t-1]*dt)*W_SV[:,t]
        volatility = vt[:,t] if SV else sigma**2
        # Simulate log asset prices
        Y_paths[:,t] = Y_paths[:,t-1] + ((mu - 0.5*volatility)*dt + np.sqrt(volatility*dt)*W1[:,t])

    Y_paths = np.exp(Y_paths)

    """ Generate Bond Data """
    B = np.exp(r*np.linspace(0,T, n_time_steps_Y))
    B = np.broadcast_to(B, Y_paths.shape)

    Payoff_Y     = np.where(Y_paths[:,-1] > K, Y_paths[:,-1], K)
    E_Payoff_Y   = Payoff_Y.mean()
    out_of_money_P  = np.where(Y_paths[:,-1] < Y, 1.0, 0.0).mean()

    """ ---------------------------------------------------------------- N Paths ---------------------------------------------------------------- """
    n_time_steps_N = int(params['T']/params['dt_n'])+1 #; print(f'Number of steps: {n_time_steps}')

    W2      = sobol_norm(params['n_paths_N'], d=n_time_steps_N)
    L_paths = np.empty((2**params['n_paths_N'], n_time_steps_N))
    L_paths[:,0] = params['l0']
    for t in range(1,n_time_steps_N):
        L_paths[:,t] = L_paths[:,t-1] + (params['c'] * L_paths[:,t-1] * params['dt'] + params['ita'] * np.sqrt(params['dt'])*W2[:,t]).squeeze()

    N_paths = np.empty((2**params['n_paths_N'], n_time_steps_N), dtype=int)
    N_paths[:,0] = params['N']
    """ Binomial Distribution of N(t) """
    for t in range(1,n_time_steps_N):
        probabilities = np.exp(-L_paths[:,t]*params['dt'])
        np.random.seed(1234+t) # to ensure reproducible results
        N_paths[:,t]  = np.random.binomial(N_paths[:,t-1], probabilities)

    S = Payoff_Y * ( np.quantile(N_paths[:,0], 0.99) / params['N'] )
    
    """ ---------------------------------------------------------------- NN ---------------------------------------------------------------- """
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
    const_initalizer = keras.initializers.RandomNormal(mean=[1-out_of_money_P,out_of_money_P], stddev=0.0, seed=1234)

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

    values          = np.empty((S.shape[0], Y_paths.shape[1]))
    values[:,-1]    = S

    """ Calculate: Payoff for Options """
    Flag = True ; its = 0
    Phi_Psi_HV  = []
    for t_i in trange(n_time_steps_Y-2, -1, -1):
        _Y_t  = Y_paths[:,t_i]
        _B_t  = B[:,t_i]
        _Y_t1 = Y_paths[:,t_i+1]
        _B_t1 = B[:,t_i+1]

        X0 = [_Y_t, np.stack((_Y_t, _B_t), axis=-1)]
        X1 = [_Y_t, np.stack((_Y_t1, _B_t1), axis=-1)]

        epochs = 300
        if Flag :
            callabacks = [rl_scheduler, keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)]
            Flag = False 
        else : 
            epochs = 20 
            callabacks = [callback]

        model.fit(X1, values[:,t_i+1], epochs=epochs, validation_split=0.0, verbose=0, batch_size=512, callbacks=callabacks) #, initial_epoch= 200 if t_i != n_time_steps-2 else 0)
        
        """ Get mean phi-psi """
        phi, psi = get_phi_psi(model, X0, mean=False)
        for f  in phi: Phi_Psi_HV.append([f,  t_i*dt, 'Phi'])
        for ps in psi: Phi_Psi_HV.append([ps, t_i*dt, 'Psi'])

        values[:,t_i] = model.predict(X0, verbose=0, batch_size=512).squeeze()

    V0 = values[:,0].mean() * ADJUSTMENT_FACTOR
    Phi = phi[0]
    Psi = psi[0]
    return V0, phi[0], psi[0]

