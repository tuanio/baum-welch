import numpy as np
from bkt import BKT
import sys
import matplotlib.pyplot as plt
plt.style.use('ggplot')

nrows = 3
ncols = 3

# save txt
# o = np.random.binomial(1, 0.5, (9, 40))
# np.savetxt('datasets.txt', o, fmt='%d')

o = np.loadtxt('datasets.txt', dtype='int')

T = o.shape[1]

model = BKT(n_iter=100, random_state=10)
model.fit(o)


def doing(idx, ax):
    data = model.plot_posterior(idx)
    p_init = data['PI'][1]
    p_transit = data['A'][1, 1]
    p_slip, p_guess = np.diag(data['B'])
    p_c = data['p_c']

    x = np.arange(T + 1)
    y = data['L']
    color = np.where(o[idx] == 1, 'red', 'blue')
    tmp = np.array(['orange'])
    color = np.concatenate((tmp, color))
    ax.scatter(x, y, c=color, s=50)
    ax.set_ylim([-0.05, 1.2])
    ax.plot(x, y)
    ax.set_title('$P(L_0)=%.2f, P(T)=%.2f, P(S)=%.2f, P(G)=%.2f, P(C_{t+1})=%.2f$' % (
        p_init, p_transit, p_slip, p_guess, p_c), fontsize=10)
    return ax

def plot_prob_obs(idx, ax):
    data = model.plot_posterior(idx)
    x = data['prob_obs']
    y = - np.log(x)
    x = np.arange(x.shape[0])
    # ax.set_ylim([-0.5, 1.2])
    ax.plot(x, y)
    ax.set_xlabel('$termination$')
    ax.set_ylabel('$-log(termination)$')
    return ax

# fig, ax = plt.subplots()
# plot_prob_obs(0, ax)
# fig.tight_layout()
# plt.show()

# plot prob training observation
fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
idx = 0
for i in range(nrows):
    for j in range(ncols):
        plot_prob_obs(idx, ax[i, j])
        idx += 1
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
idx = 0
for i in range(nrows):
    for j in range(ncols):
        doing(idx, ax[i, j])
        idx += 1
fig.tight_layout()
plt.show()

