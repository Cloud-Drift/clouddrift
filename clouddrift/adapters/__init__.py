"""
This module provides adapters to custom datasets.
Each adapter module provides convenience functions and metadata to convert a
custom dataset to a `clouddrift.RaggedArray` instance.
Currently, clouddrift provides adapter modules for the hourly Global
Drifter Program (GDP) dataset, the 6-hourly GDP dataset, 15-minute Grand LAgrangian
Deployment (GLAD) dataset, and the MOSAiC sea-ice drift dataset. More adapters will be added
in the future.
"""

import clouddrift.adapters.andro
import clouddrift.adapters.gdp1h
import clouddrift.adapters.gdp6h
import clouddrift.adapters.glad
import clouddrift.adapters.mosaic
import clouddrift.adapters.subsurface_floats
import clouddrift.adapters.yomaha
