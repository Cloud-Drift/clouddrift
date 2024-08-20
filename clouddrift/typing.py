from datetime import timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

# Subscripting the type for pandas series only works at type checking time
if TYPE_CHECKING:
    _SupportedArrayTypes = (
        list[Any] | np.ndarray[Any, np.dtype[Any]] | xr.DataArray | pd.Series[Any]
    )
else:
    _SupportedArrayTypes = (
        list[Any] | np.ndarray[Any, np.dtype[Any]] | xr.DataArray | pd.Series
    )

_ArrayTypes = _SupportedArrayTypes

_SupportedTimeDeltaTypes = pd.Timedelta | timedelta | np.timedelta64
_TimeDeltaTypes = _SupportedTimeDeltaTypes

__all__ = ["_ArrayTypes", "_TimeDeltaTypes"]
