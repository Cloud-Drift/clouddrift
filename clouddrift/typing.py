import datetime
import typing

import numpy as np
import numpy.typing as np_typing
import pandas as pd
import xarray as xr

# Subscripting the type for pandas series only works at type checking time
if typing.TYPE_CHECKING:
    _SupportedArrayTypes = (
        list[typing.Any]
        | np_typing.NDArray[typing.Any]
        | pd.Series[typing.Any]
        | xr.DataArray
    )
else:
    _SupportedArrayTypes = (
        list[typing.Any] | np_typing.NDArray[typing.Any] | pd.Series | xr.DataArray
    )

ArrayTypes: typing.TypeAlias = _SupportedArrayTypes

_SupportedToleranceTypes = pd.Timedelta | datetime.timedelta | np.timedelta64
ToleranceTypes: typing.TypeAlias = _SupportedToleranceTypes

__all__ = ["ArrayTypes", "ToleranceTypes"]
