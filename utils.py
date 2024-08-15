import numpy as np


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def averaging_reward(x):
    cumsum = np.cumsum(x)
    return cumsum / np.arange(1, len(x) + 1)
