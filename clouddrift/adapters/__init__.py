"""
This module provides adapters to custom datasets.
Each adapter module provides convenience functions and metadata to convert a
custom dataset to a `clouddrift.RaggedArray` instance.
Currently, clouddrift only provides adapter modules for the hourly Global
Drifter Program (GDP) data, 6-hourly GDP data, and the MOSAiC sea-ice drift
data. More adapters will be added in the future.
"""

import clouddrift.adapters.gdp1h
import clouddrift.adapters.gdp6h
import clouddrift.adapters.mosaic
