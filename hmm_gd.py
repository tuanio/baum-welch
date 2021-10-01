import numpy as np
from algorithms import forward, backward
from algorithms import hmmlearn_model


def cal_xi(t, i, j):
    numerator = alpha[i, t] * A[i, j] * beta[j, t + 1] * B[j, o[t + 1]]
    denominator = 0
    for i in range(N):
        for j in range(N):
            denominator += alpha[i, t] * A[i, j] * \
                beta[j, t + 1] * B[j, o[t + 1]]
    return numerator / denominator


def cal_gamma(t, j):
    return alpha[j, t]*beta[j, t]/(alpha[:, t] * beta[:, t]).sum()


# o = np.loadtxt('datasets.txt', dtype='int')[0]
o = np.array([0, 1, 1, 0, 0])
T = o.shape[0]
N = 2
p_init = 0.5
p_transit = 0.1
p_slip = 0.1
p_guess = 0.1
PI = np.array([1 - p_init, p_init])
A = np.array([[0, 1], [1 - p_transit, p_transit]])
B = np.array([[p_slip, 1 - p_slip], [1 - p_guess, p_guess]])
alpha = forward(PI, A, B, o)[0]
beta = backward(PI, A, B, o)[0]

gamma = alpha * beta
gamma /= gamma.sum(axis=0)

xi = np.zeros((T, N, N))
for t in range(T - 1):
    for i in range(N):
        xi[t, i, :] = alpha[i, t] * beta[:, t + 1] * A[i, :] * B[:, o[t + 1]]
    xi[t, :, :] /= xi[t, :, :].sum()

PI_est = gamma[:, 0]
A_est = xi[:-1, :, :].sum(axis=0) / gamma[:, :-1].sum(axis=1)[:, np.newaxis]
B_est = np.zeros_like(B)
B_est[:, 0] = gamma[:, o == 0].sum(axis=1) / gamma.sum(axis=1)
B_est[:, 1] = gamma[:, o == 1].sum(axis=1) / gamma.sum(axis=1)
