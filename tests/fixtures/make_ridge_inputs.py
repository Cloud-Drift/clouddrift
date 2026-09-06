"""Write the inputs that both CloudDrift and jLab analyse, for the regression fixtures.

Run from the repository root::

    python tests/fixtures/make_ridge_inputs.py

It writes ``data/fixtures/ridge_fixture_inputs.mat``. MATLAB then reads that file, runs
jLab, and writes ``ridge_fixture_outputs.mat`` beside it; see ``make_ridge_fixtures.m``.

Generating the inputs on the Python side, rather than in each language separately, is what
makes the comparison meaningful: both sides analyse identical numbers, including the same
frequency grid, so any disagreement is in the algorithms rather than in the setup.

``data/`` is gitignored. The inputs are reproducible from this script, and the outputs are
converted into a compact ``.npz`` for the test suite rather than committed as MATLAB files.
"""

import numpy as np
import xarray as xr
from scipy.io import savemat

from clouddrift.wavelet import morse_logspace_freq

GAMMA, BETA = 3.0, 2.0
DRIFTER_FILE = "data/gdp1h_44000.nc"
OUTPUT_FILE = "data/fixtures/ridge_fixture_inputs.mat"
# MATLAB datenum counts days from year 0; the Unix epoch is day 719529
DATENUM_EPOCH = 719529.0


def synthetic_signal(n_times=4096, seed=20260905):
    """A modulated ellipse, a faster counter-rotating oscillation, and a red background."""
    rng = np.random.default_rng(seed)
    time = np.arange(n_times, dtype=float)
    eddy = 40.0 * np.exp(1j * 2 * np.pi * time / 400.0) * ((time > 1000) & (time < 3000))
    fast = 4.0 * np.exp(-1j * 2 * np.pi * time / 25.0)
    background = np.cumsum(rng.normal(size=n_times) + 1j * rng.normal(size=n_times)) * 0.5
    return eddy + fast + background


def main():
    signal = synthetic_signal()
    frequencies = morse_logspace_freq(GAMMA, BETA, signal.size, density=4)

    drifter = xr.open_dataset(DRIFTER_FILE)
    latitude = drifter["lat"].values.astype(float)
    longitude = drifter["lon"].values.astype(float)
    datenum = drifter["time"].values.astype("datetime64[s]").astype(float) / 86400.0 + DATENUM_EPOCH

    savemat(
        OUTPUT_FILE,
        {
            # case 1 and 2: the synthetic signal, in sample units
            "x": np.real(signal)[:, None],
            "y": np.imag(signal)[:, None],
            "fs": frequencies[:, None],
            "dt": 1.0,
            # case 3: the drifter
            "lat": latitude[:, None],
            "lon": longitude[:, None],
            "num": datenum[:, None],
            # parameters, matching the settings of Lilly and Perez-Brunius (2021)
            "ga": GAMMA,
            "be": BETA,
            "P": np.sqrt(GAMMA * BETA),
            "fmax": 2.0,
            "fmin": 1.0 / 64.0,
            "M": 1.0,
            "RHO": 0.0,
        },
        do_compression=True,
    )
    print(f"wrote {OUTPUT_FILE}")
    print(f"  synthetic: {signal.size} samples, {frequencies.size} frequencies")
    print(f"  drifter:   {latitude.size} hourly points")


if __name__ == "__main__":
    main()
