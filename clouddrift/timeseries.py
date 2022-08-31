import numpy as np


def periodogram(uv):
    """Calculate the periodogram of the rotary velocity

    Args:
        uv (ak.Array): ragged arrays of rotary velocity

    Returns:
        ak.Array: ragged arrays of periodogram
    """

    dt = 1 / 24
    d = []
    for i in range(0, len(uv)):
        d.append(dt * np.abs(np.fft.fft(uv[i])) ** 2)
    return d
