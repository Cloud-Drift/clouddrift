import numpy as np


def periodogram(uv):
    dt = 1 / 24
    d = []
    for i in range(0, len(uv)):
        d.append(dt * np.abs(np.fft.fft(uv[i])) ** 2)
    return d
