import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

class Utils:
    def __init__(self, random_state=1):
        self.random_state = random_state
        self.rs = np.random.RandomState(random_state)

    def forward(self, init, transition, emission, o):
        T = len(o)
        N = len(transition)
        alpha = np.zeros(shape=(N, T))
        alpha[:, 0] = init * emission[:, o[0]]
        for t in range(1, T):
            alpha[:, t] = alpha[:, t -
                                1].dot(transition).dot(np.diag(emission[:, o[t]]))
        termination = alpha[:, T - 1].sum()
        return alpha, termination

    def backward(self, init, transition, emission, o):
        T = len(o)
        N = len(transition)
        beta = np.zeros(shape=(N, T))
        beta[:, T - 1] = 1
        for t in range(T - 2, -1, -1):
            beta[:, t] = transition.dot(
                np.diag(emission[:, o[t + 1]])).dot(beta[:, t + 1])
        termination = np.sum(init * emission[:, o[0]] * beta[:, 0])
        return beta, termination

    def forward_backward(self, o, n_iters=10000, threshold=1e-6, verbose=1):
        N = 2
        T = len(o)
        V = np.unique(o)
        p_init, p_transit, p_slip, p_guess = self.rs.rand(4)
        PI = np.array([1 - p_init, p_init])
        A = np.array([[0, 1], [1 - p_transit, p_transit]])
        B = np.array([[p_slip, 1 - p_slip], [1 - p_guess, p_guess]])

        alpha = self.forward(PI, A, B, o)[0]
        beta = self.backward(PI, A, B, o)[0]

        def cal_gamma(t, j):
            return alpha[j, t]*beta[j, t]/(alpha[:, t] * beta[:, t]).sum()

        def cal_xi(t, i, j):
            return alpha[i, t]*A[i, j]*B[j, o[t + 1]]*beta[j, t + 1]/(alpha[:, t]*beta[:, t]).sum()

        for _ in range(n_iters):

            PI_est = np.zeros_like(PI)
            for i in range(N):
                PI_est[i] = cal_gamma(1, i)

            A_est = np.zeros_like(A)
            for i in range(N):
                for j in range(N):
                    foo = 0
                    for t in range(T - 1):
                        A_est[i, j] += cal_xi(t, i, j)
                        for k in range(N):
                            foo += cal_xi(t, i, k)
                    A_est[i, j] /= foo

            B_est = np.zeros_like(B)
            for j in range(N):
                foo = 0
                for v in V:
                    for t in np.argwhere(o == v):
                        bar = cal_gamma(t, j)
                        B_est[j, v] += bar
                        foo += bar
                B_est[j, :] /= foo

            if np.linalg.norm(PI - PI_est) <= threshold and\
                    np.linalg.norm(A - A_est) <= threshold and \
                    np.linalg.norm(B - B_est) <= threshold:
                if verbose:
                    print('Converged at:', _ + 1)
                break

            PI = PI_est
            A = A_est
            B = B_est

        return PI, A, B


class BKT:
    def __init__(self, random_state, algorithm='self', verbose=1):
        self.data_ = []
        self.random_state = random_state
        self.algorithm = algorithm
        self.utils = Utils(random_state)
        self.verbose = verbose

    def fit(self, obs_matrix):
        self.data_ = []
        self.partial_fit(obs_matrix)
        return self

    def partial_fit(self, obs_matrix):
        for obs in obs_matrix:
            if self.algorithm == 'self':
                PI, A, B = self.utils.forward_backward(obs, verbose=self.verbose)
            else:
                model = hmm.MultinomialHMM(n_components=2, init_params='se')
                model.transmat_ = [[0, 1], [0.5, 0.5]]
                model.fit(obs.reshape(-1, 1))
                PI, A, B = model.startprob_, model.transmat_, model.emissionprob_
            L, p_c = self.cal_L(PI, A, B, obs)
            data = dict(PI=PI, A=A, B=B, L=L, p_c=p_c)
            self.data_.append(data)
        return self

    def cal_L(self, PI, A, B, obs):
        p_init = PI[1]
        p_transit = A[1, 1]
        p_slip, p_guess = np.diag(B)
        L = [p_init]
        for o in obs:
            if o:
                p_l = L[-1]*(1-p_slip)/(L[-1]*(1-p_slip)+(1-L[-1])*p_guess)
            else:
                p_l = L[-1]*p_slip/(L[-1]*p_slip+(1-L[-1])*(1-p_guess))
            p_l = p_l + (1 - p_l)*p_transit
            L.append(p_l)
        p_c = L[-1]*(1-p_slip) + (1-L[-1])*p_guess
        return L, p_c

    def plot_posterior(self, idx, y):
        data = self.data_[idx]
        y = np.concatenate((np.array([1]), y))
        x = np.arange(len(data['L']))
        plt.plot(x, data['L'])
        plt.scatter(x, data['L'], c=np.where(y==1, 'red', 'blue'))
        plt.xlabel('Time')
        plt.ylabel('$P(L=1)$')
        plt.ylim([0, 1.2])
        plt.show()