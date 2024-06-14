---
title: 'Clouddrift: a Python package to accelerate the use of Lagrangian data for atmospheric, oceanic, and climate sciences'
tags:
  - Python
  - climate
  - ocean
  - atmosphere
  - ragged array
authors:
  - name: Shane Elipot
    orcid: 0000-0001-6051-5426
    equal-contrib: true
    affiliation: 1
  - name: Philippe Miron
    orcid: 0000-0002-8520-6221
    equal-contrib: true
    affiliation: 2
  - name: Milan Curcic
    orcid: 0000-0002-8822-7749
    equal-contrib: true
    affiliation: 1, 4
  - name: Kevin Santana
    orcid: 0009-0003-8383-1212
    equal-contrib: true
    affiliation: 1
  - name: Rick Lumpkin
    orcid: 0000-0002-6690-1704
    equal-contrib: true
    affiliation: 3

affiliations:
 - name: Rosenstiel School of Marine, Atmospheric, and Earth Science, University of Miami
   index: 1
 - name: Florida State University
   index: 2
 - name: NOAA Atlantic Oceanographic and Meteorological Laboratory
   index: 3
 - name: Frost Institute for Data Science and Computing, University of Miami
   index: 4
date: 10 April 2024
bibliography: paper.bib

---
 
# Summary

Lagrangian data in Earth sciences are unique because they do not conform to established standards related to dimensions, coordinates, and organizational structures. In addition, because they convolve spatial and temporal information, Lagrangian data need specific processing and analysis tools for their scientific and operational use. The clouddrift Python library addresses these challenges by offering tools to process and analyze Lagrangian data with an emphasis on the ragged array representation.

# Statement of need

In Earth, Ocean, Geo-, and Atmospheric Science, *Eulerian data* typically refers to a type of data acquired or simulated at a particular fixed point or region in space. Eulerian data are defined on fixed spatiotemporal grids with monotonic coordinates (e.g. latitude, longitude, depth, time) for which popular Python tools such as [Xarray](https://docs.xarray.dev/en/stable/) [@Hoyer:2017] are naturally suited. In contrast, *Lagrangian data* are acquired by observing platforms that move with the flow they are embedded in, for example, uncrewed platforms, vehicles, virtual particles, atmospheric phenomena such as tropical cyclones, and even animals that gather data along their natural but complex paths. Because such paths traverse both spatial and temporal dimensions, Lagrangian data often convolve spatial and temporal information that cannot consistently and readily be organized, cataloged, and stored in common data structures and file formats with the help of common libraries and standards. As an example, the concepts of dimensions and coordinates for Lagrangian data are ambiguous and not clearly established. As such, for both data generators and data users, Lagrangian data present challenges that the clouddrift Python library aims to overcome. 

The clouddrift library is distinct from other tools designed to simulate particle trajectories in oceanic and atmospheric models, such as [OceanParcels](https://oceanparcels.org) [@Delandmeter:2019], or [HYSPLIT](https://www.ready.noaa.gov/HYSPLIT.php) [@Stein:2015]. Unlike these tools, clouddrift focuses on analyzing data from observational and numerical Lagrangian experiments and curating Lagrangian datasets using consistent data structures and methodologies ([See Scope and key features](#scope-and-key-features)). While clouddrift shares some goals with argopy [@Maze:2020], a Python library for accessing and manipulating the Argo dataset (a specific Lagrangian oceanographic dataset), clouddrift aims to be dataset-agnostic and extends beyond just Earth data. Additionally, clouddrift incorporates oceanographic analysis functions from jLab, a Matlab data analysis package [@Lilly:2021], in compliance with its license. Clouddrift core Python dependencies include NumPy [@Harris:2020] and SciPy [@Virtanen:2020] for data analysis, as well as Xarray [@Hoyer:2017], pandas [@reback2020pandas;@mckinney-proc-scipy-2010], and [Awkward Array](https://awkward-array.org/doc/main/index.html) for its data processing and manipulation functions. 

# Scope and key features

The scope of the clouddrift library includes: 

1. **Working with contiguous ragged array representations of data, whether they originate from geosciences or any other field**. Ragged array representations are useful when the data lengths of the instances of a feature (variable) are not all equal. With such representations the data for each feature are stored contiguously in memory, and the number of elements that each feature has is contained in a count variable which clouddrift calls *rowsize*. A graphical representation of the application of the ragged array structure to Lagrangian data is displayed in \autoref{fig:raggedarray}.

![Ragged array representation for Lagrangian data. \label{fig:raggedarray}](ragged_array.png)

2. **Delivering functions and methods to perform scientific analysis of Lagrangian data, oceanographic or otherwise [@Lacasce:2008;@Vansebille:2018], structured as ragged arrays or otherwise**. A straightforward example of Lagrangian analysis provided by clouddrift is the derivation of Lagrangian velocities from a sequence of Lagrangian positions, and vice versa. Another more involved example is the discovery of pairs of Lagrangian data prescribed by distances in space and time. Both of these methods are currently available with clouddrift.

*Example*: The following example illustrates how to combine two functions from the clouddrift library in order to calculate Lagrangian velocities from ragged arrays of Cartesian positions and times that share row sizes 2, 3, and 4:
```python
import numpy as np
from clouddrift.kinematics import velocity_from_position
from clouddrift.ragged import apply_ragged

rowsize = [2, 3, 4]
x = np.array([1, 2, 10, 12, 14, 30, 33, 36, 39])
y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
t = np.array([1, 2, 1, 2, 3, 1, 2, 3, 4])

u1, v1 = apply_ragged(velocity_from_position, [x, y, t], rowsize,
  coord_system="cartesian")
```

3. **Processing publicly available Lagrangian datasets into the common ragged array data structure and format**. Through data *adapters*, this type of processing includes not only converting Lagrangian data from typically regular arrays to ragged arrays but also aggregating data and metadata from multiple data files into a single data file. The canonical example of the clouddrift library is constituted of the data from the NOAA Global Drifter Program [@Elipot:2022].

*Example:* The following example locally builds an xarray dataset, with ragged array representations, of the latest dataset of position, velocity, and sea surface temperature from the Global Drifter Program quality-controlled 6-hour interpolated data from ocean surface drifting buoys:
```python
from clouddrift.adapters import gdp6h
ds = gdp6h.to_raggedarray().to_xarray()
```

4. **Making cloud-optimized ragged array datasets easily accessible**. This involves opening in a computing environment, without unnecessary download, Lagrangian datasets available from cloud servers, as well as opening Lagrangian datasets that have been seamlessly processed by the clouddrift data *adapters*.    

*Example:* The following simple command remotely opens without downloading the hourly location, current velocity, and temperature collected from Global Drifter Program drifters worldwide, distributed as a Zarr archive with ragged array representations and stored in cloud storage as part of the [Registry of Open Data on AWS](https://registry.opendata.aws/noaa-oar-hourly-gdp/):

```python
from clouddrift.datasets import gdp1h
ds = gdp1h()
```

# Acknowledgements

The development of the clouddrift library is a result of [NSF Award #2126413: *EarthCube Capabilities: CloudDrift: a platform for accelerating research with Lagrangian climate data*](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2126413). SE, PM, MC, and KS have been partially supported by this award. 

# References
