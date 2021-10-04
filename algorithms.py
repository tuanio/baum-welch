import numpy as np
from hmmlearn import hmm
from pprint import pprint


def cost(x):
    return - np.log(x)


def forward(init, transition, emission, o):
    T = len(o)
    N = len(transition)
    alpha = np.zeros(shape=(N, T))
    alpha[:, 0] = init * emission[:, o[0]]
    for t in range(1, T):
        alpha[:, t] = alpha[:, t -
                            1].dot(transition).dot(np.diag(emission[:, o[t]]))
    termination = alpha[:, T - 1].sum()
    return alpha, termination


def backward(init, transition, emission, o):
    T = len(o)
    N = len(transition)
    beta = np.zeros(shape=(N, T))
    beta[:, T - 1] = 1
    for t in range(T - 2, -1, -1):
        beta[:, t] = transition.dot(
            np.diag(emission[:, o[t + 1]])).dot(beta[:, t + 1])
    termination = np.sum(init * emission[:, o[0]] * beta[:, 0])
    return beta, termination


def estimate_posterior(p_init, p_transit, p_slip, p_guess, o, n_iter, threshold):
    N = 2
    T = len(o)
    V = np.unique(o)  # vocabulary
    PI = np.array([1 - p_init, p_init])
    A = np.array([[0, 1], [1 - p_transit, p_transit]])
    B = np.array([[p_slip, 1 - p_slip], [1 - p_guess, p_guess]])
    prob_obs = []

    alpha, termination = forward(PI, A, B, o)
    beta = backward(PI, A, B, o)[0]
    prob_obs.append(termination)

    for _ in range(n_iter):

        gamma = alpha * beta
        gamma /= gamma.sum(axis=0)

        xi = np.zeros((T, N, N))
        for t in range(T - 1):
            for i in range(N):
                xi[t, i, :] = alpha[i, t] * \
                    beta[:, t + 1] * A[i, :] * B[:, o[t + 1]]
            xi[t, :, :] /= xi[t, :, :].sum()

        PI_est = gamma[:, 0]
        A_est = xi[:-1, :, :].sum(axis=0) / \
            gamma[:, :-1].sum(axis=1)[:, np.newaxis]
        B_est = np.zeros_like(B)
        for v in V:
            B_est[:, v] = gamma[:, o == v].sum(axis=1)
        B_est /= gamma.sum(axis=1)[:, np.newaxis]

        alpha, termination = forward(PI_est, A_est, B_est, o)
        beta = backward(PI_est, A_est, B_est, o)[0]
        prob_obs.append(termination)

        if np.linalg.norm(A - A_est) <= threshold and np.linalg.norm(B - B_est) <= threshold and np.linalg.norm(PI - PI_est) <= threshold:
            loss = -np.log(termination)
            print(
                f"Converged at: {_}. P(L0)={round(p_init, 2)}, P(T)={round(p_transit, 2)}, P(S)={round(p_slip, 2)}, P(G)={round(p_guess, 2)}. Terminations: {loss}")
            break

        PI = PI_est
        A = A_est
        B = B_est

    return PI, A, B, termination, prob_obs


def baum_welch(o, p_init=0.5, p_transit=0.1, p_slip=0.1, p_guess=0.1, n_iter=100, threshold=1e-2, verbose=1):
    probs = np.linspace(0, 1, 20, endpoint=False)[1:]
    # probs = np.linspace(0.1, 0.9, 9)
    # 0.1, 0.2, 0.3, ... 0.9
    loss = float('inf')
    idx = 0
    best_init_params = {}
    for p_init in probs:
        for p_transit in probs:
            for p_slip in probs:
                for p_guess in probs:
                    idx += 1
                    PI, A, B, termination, prob_obs = estimate_posterior(
                        p_init, p_transit, p_slip, p_guess, o, n_iter, threshold)
                    now_loss = -np.log(termination)
                    if now_loss < loss:
                        loss = now_loss
                        PI_best = PI
                        A_best = A
                        B_best = B
                        prob_obs_best = prob_obs
                        best_init_params = {
                            'p_init': p_init,
                            'p_transit': p_transit,
                            'p_slip': p_slip,
                            'p_guess': p_guess
                        }

    return PI_best, A_best, B_best, np.array(prob_obs_best), best_init_params


def hmmlearn_model(obs, n_iter=100, random_state=42):
    model = hmm.MultinomialHMM(
        n_components=2, init_params='', n_iter=n_iter, random_state=random_state)
    model.startprob_ = [0.5, 0.5]
    model.transmat_ = [[0, 1], [0.9, 0.1]]
    model.emissionprob_ = [[0.1, 0.9], [0.9, 0.1]]
    model.fit(obs)
    return model.startprob_, model.transmat_, model.emissionprob_
