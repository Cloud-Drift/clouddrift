from ._version import version

__version__ = version

import clouddrift.adapters as adapters
import clouddrift.datasets as datasets
import clouddrift.kinematics as kinematics
import clouddrift.pairs as pairs
import clouddrift.plotting as plotting
import clouddrift.ragged as ragged
import clouddrift.signal as signal
import clouddrift.sphere as sphere
import clouddrift.wavelet as wavelet
from clouddrift.raggedarray import RaggedArray

__all__ = [
    "RaggedArray",
    "adapters",
    "datasets",
    "kinematics",
    "pairs",
    "plotting",
    "ragged",
    "signal",
    "sphere",
    "wavelet",
]
