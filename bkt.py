import numpy as np
import matplotlib.pyplot as plt
from algorithms import baum_welch, hmmlearn_model
import pickle


class BKT:
    def __init__(self, n_iter=100, random_state=42, verbose=1):
        self.data = []
        self.random_state = random_state
        self.n_iter = n_iter
        self.verbose = verbose

    def fit(self, obs_matrix):
        self.data = []
        self.obs_matrix = obs_matrix
        self.partial_fit(obs_matrix)
        return self

    def partial_fit(self, obs_matrix):
        for idx, obs in enumerate(obs_matrix):
            print(f'Idx: {idx}', end=': ')
            PI, A, B, prob_obs, init_params = baum_welch(
                obs, n_iter=self.n_iter, verbose=self.verbose)
            L, p_c = self.cal_L(PI, A, B, obs)
            data = dict(prob_obs=prob_obs, PI=PI, A=A, B=B,
                        L=L, p_c=p_c, init_params=init_params)
            self.data.append(data)
        # save best model
        pickle.dump(self.data, open('best_params.pkl', 'wb'))
        return self

    def cal_L(self, PI, A, B, obs):
        p_init = PI[1]
        p_transit = A[1, 1]
        p_slip, p_guess = np.diag(B)
        L = [p_init]
        for o in obs:
            L.append(p_init)
            if o:
                p_l = L[-1]*(1-p_slip)/(L[-1]*(1-p_slip)+(1-L[-1])*p_guess)
            else:
                p_l = L[-1]*p_slip/(L[-1]*p_slip+(1-L[-1])*(1-p_guess))
            p_l = p_l + (1 - p_l)*p_transit
            L[-1] = p_l
            p_init = p_l
        p_c = L[-1]*(1-p_slip) + (1-L[-1])*p_guess
        return L, p_c

    def plot_posterior(self, idx):
        return self.data[idx]
        x = np.arange(len(data['L']))
        y = self.obs_matrix[idx]
        # plt.plot(x, data['L'])
        # plt.scatter(x, data['L'], c=np.where(y==1, 'red', 'blue'))
        # plt.xlabel('Time')
        # plt.ylabel('$P(L=1)$')
        # # plt.ylim([0, 1.2])
        # plt.show()
