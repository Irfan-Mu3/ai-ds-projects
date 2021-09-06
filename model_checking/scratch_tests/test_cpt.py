import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

"""
Playing with propect theory.
Proves non-linearity of cp. i.e. CP[X+Y] != CP[X] + CP[Y]
"""


def cp(prob_vec, gamma):
    a = prob_vec ** gamma
    b = (1 - prob_vec) ** gamma

    return a / (a + b) ** (1 / gamma)


if __name__ == '__main__':
    N = 1000
    xs = np.arange(N)

    prob_X = st.binom(N / 4, 0.3).pmf(xs)
    prob_Y = st.binom(N / 2, 0.2).pmf(xs)
    probX_Y = np.convolve(prob_X, prob_Y)

    plt.plot(prob_X, label='X')
    plt.plot(prob_Y, label='Y')
    plt.plot(probX_Y, label='X+Y')

    gamma = 0.7
    cpX = cp(prob_X, gamma)
    cpY = cp(prob_Y, gamma)
    cpX_Y = cp(probX_Y, gamma)

    print("sums to 1?:",np.sum(cpX),np.sum(cpY),np.sum(cpX_Y))

    print("linear average", (xs * prob_X).sum() + (prob_Y * xs).sum())
    print(" truth average", (np.arange((2 * N) - 1) * probX_Y).sum())

    print("linear cpt", (xs * cpX).sum() + (cpY * xs).sum())
    print(" truth cpt", (np.arange((2 * N) - 1) * cpX_Y).sum())

    plt.plot(cpX * xs, label='X')
    plt.plot(cpY * xs, label='Y')
    plt.plot(cpX * xs + cpY * xs, label='linear?')
    plt.plot(cpX_Y * np.arange((2 * N) - 1), label='truth')
    plt.legend()
    plt.show()
