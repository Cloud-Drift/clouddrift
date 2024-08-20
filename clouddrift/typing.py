from datetime import timedelta
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import xarray as xr

T = TypeVar("T")
_SupportedArrayTypes = (
    list[Any] | np.ndarray[Any, np.dtype[Any]] | xr.DataArray | pd.Series[Any]
)
_ArrayTypes = _SupportedArrayTypes

_SupportedTimeDeltaTypes = pd.Timedelta | timedelta | np.timedelta64
_TimeDeltaTypes = _SupportedTimeDeltaTypes

__all__ = ["_ArrayTypes", "_TimeDeltaTypes"]
