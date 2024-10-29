import datetime
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

# Subscripting the type for pandas series only works at type checking time
if TYPE_CHECKING:
    _SupportedArrayTypes = list[Any] | NDArray[Any] | pd.Series[Any] | xr.DataArray
else:
    _SupportedArrayTypes = list[Any] | NDArray[Any] | pd.Series | xr.DataArray

ArrayTypes: TypeAlias = _SupportedArrayTypes

_SupportedToleranceTypes = pd.Timedelta | datetime.timedelta | np.timedelta64
ToleranceTypes: TypeAlias = _SupportedToleranceTypes

__all__ = ["ArrayTypes", "ToleranceTypes"]
