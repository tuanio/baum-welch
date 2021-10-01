import numpy as np
from algorithms import forward, backward

p_init = 0.5
p_transit = 0.1
p_slip = 0.1
p_guess = 0.1

# o = np.random.binomial(1, 0.5, (1, 2))
# np.savetxt('testset.txt', o, fmt='%d')
o = np.loadtxt('testset.txt', dtype='int')

# PI = np.array([1 - p_init, p_init])
# A = np.array([[0, 1], [1 - p_transit, p_transit]])
# B = np.array([[p_slip, 1 - p_slip], [1 - p_guess, p_guess]])

PI = np.array([1, 2])
A = np.array([[3, 4], [2, 1]])
B = np.array([[3, 2], [3, 5]])

alpha = forward(PI, A, B, o)[0]
beta = backward(PI, A, B, o)[0]

data = alpha * beta
data /= data.sum(axis=0)
print(alpha)
print(beta)
print(data)