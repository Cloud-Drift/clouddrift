"""
This module defines functions used to adapt the YoMaHa dataset as
a ragged-array dataset. 

The dataset is hosted at http://apdrc.soest.hawaii.edu/projects/yomaha/

Example
-------
>>> from clouddrift.adapters import yomaha
>>> ds = yomaha.to_xarray()
"""

from datetime import datetime
import gzip
import numpy as np
import os
import pandas as pd
import tempfile
from tqdm import tqdm
import urllib.request
import xarray as xr
import warnings


# order of the URLs is important
YOMAHA_URLS = [
    "http://apdrc.soest.hawaii.edu/projects/Argo/data/trjctry/float_types.txt",
    "http://apdrc.soest.hawaii.edu/projects/Argo/data/trjctry/DACs.txt",
    "http://apdrc.soest.hawaii.edu/projects/Argo/data/trjctry/0-Near-Real_Time/0-date_time.txt",
    "http://apdrc.soest.hawaii.edu/projects/Argo/data/trjctry/0-Near-Real_Time/WMO2DAC2type.txt",
    "http://apdrc.soest.hawaii.edu/projects/Argo/data/trjctry/0-Near-Real_Time/end-prog.lst",
    "http://apdrc.soest.hawaii.edu/projects/Argo/data/trjctry/0-Near-Real_Time/yomaha07.dat.gz",
]

YOMAHA_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "yomaha")


# Create a function to download a file with a progress bar
def download_with_progress(url, output_file):
    # Check if the file already exists
    if os.path.isfile(output_file):
        # Get last modified time of the local file
        local_last_modified = os.path.getmtime(output_file)

        # Make a HEAD request to get remote file info
        with urllib.request.urlopen(url) as response:
            remote_last_modified = datetime.strptime(
                response.headers.get("Last-Modified"), "%a, %d %b %Y %H:%M:%S %Z"
            )

            # Compare last modified times
            if local_last_modified >= remote_last_modified.timestamp():
                warnings.warn(
                    f"{output_file} already exists and is up to date; skip download."
                )
                return False

    with urllib.request.urlopen(url) as response, open(
        output_file, "wb"
    ) as outfile, tqdm(
        desc=url,
        total=int(response.headers["Content-Length"] or 0),
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        chunk_size = 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            outfile.write(chunk)
            bar.update(len(chunk))
    return True


def download(tmp_path: str):
    # loop on different files to download
    for i in range(0, len(YOMAHA_URLS) - 1):
        print("Downloading: " + str(YOMAHA_URLS[i]))
        outfile = tmp_path + YOMAHA_URLS[i].split("/")[-1]
        download_with_progress(YOMAHA_URLS[i], outfile)

    ## gzip file (saved archive ~120Mb and decompressed ~400Mb)
    print("Downloading: " + str(YOMAHA_URLS[-1]))
    filename_gz = tmp_path + YOMAHA_URLS[-1].split("/")[-1]

    if download_with_progress(YOMAHA_URLS[-1], filename_gz) or not os.path.isfile(
        filename := filename_gz[:-3]
    ):
        with open(filename_gz, "rb") as f_gz, open(filename, "wb") as f:
            f.write(gzip.decompress(f_gz.read()))


def to_xarray(tmp_path: str = None):
    if tmp_path is None:
        tmp_path = YOMAHA_TMP_PATH
        os.makedirs(tmp_path, exist_ok=True)

    # get or update required files
    download(tmp_path)

    # database last update
    with open(tmp_path + YOMAHA_URLS[2].split("/")[-1]) as f:
        print("Last database update was: " + f.read())

    """
    ## Columns 1-8 contain three-dimensional coordinates, time, components and errors of the deep float velocity. 
    - 1-2. Coordinates (longitude Xndeep and latitude Yndeep) of location where deep velocity is estimated. These coordinates are averages between last fixed float position (Xn-1last , Yn-1last) at the sea surface during previous cycle (stored in columns 16-17) and first fix (Xnfirst , Ynfirst) in the current cycle (stored in columns 19-20).  I.e., [X,Y] ndeep = ( [X,Y] n-1last + [X,Y] nfirst)/2.
    - 3. “Parking” pressure Zpark (dbars) for this cycle. This value is a pre-programmed value stored in the “meta”-file. 
    - 4. Julian time Tndeep (days) relative to 2000-01-01 00:00 UTC. (Adding 18262 will convert it into more traditional Julian time relative to 1950-01-01 00:00 UTC.) This value is an average between the Julian time of the last fix during the previous cycle (Tn-1last  stored in column 18) and the first fix in the current cycle (Tnfirst  stored in column 21). I.e., Tndeep = ( Tn-1last + Tnfirst)/2 . 
    - 5-6. Estimate of eastward and northward components of the deep velocity (Undeep , Vndeep) (cm/s) at Zparkcalculated from the float displacement from  [X,Y] n-1last to [X,Y] nfirst for time Tnfirst - Tn-1last. 
    - 7-8. Estimates of the errors of components of deep velocity (εUndeep , εVndeep) (cm/s) due to a vertical shear of horizontal flow obtained as described in Appendix A. 
    ## Columns 9-15 contain horizontal coordinates, time, components and errors of the float velocity at the sea surface. Velocity is estimated using linear regression of all surface fixes for the cycle. Details are given in Appendix B. 
    - 9-10. Coordinates (longitude Xnsurf and latitude Ynsurf) of location where surface velocity is estimated. 
    - 11. Julian time Tnsurf (days) relative to 2000-01-01 00:00 UTC when surface velocity is estimated. 
    - 12-13. Estimate of eastward and northward components of velocity (Unsurf , Vnsurf) (cm/s) at the sea surface. - 
    - 14-15. Estimates of the errors of components of surface velocity (εUnsurf , εVnsurf) (cm/s) obtained as described in Appendix B. 

    ## Auxiliary float and cycle data are in columns 16-27. 
    - 16-18. Coordinates (Xn-1last , Yn-1last) and Julian time Tn-1last (relative to 2000-01-01 00:00 UTC) of the last fix at the sea surface during the previous cycle. 
    - 19-21. Coordinates (Xnfirst , Ynfirst) and Julian time Tnfirst (relative to 2000-01-01 00:00 UTC)of the first fix at the sea surface during the current cycle.
    - 22-24. Coordinates (Xnlast , Ynlast) and Julian time Tnlast(relative to 2000-01-01 00:00 UTC)of the last fix at the sea surface during the current cycle. 
    - 25. Number of surface fixes Nnfix during the current cycle. 
    - 26. Float ID. To unify data of all DAC’s we re-counted all the floats. Correspondence between our float ID’s and WMO float ID’s used by the DAC’s is described in our file yomaha2wmo.dat 
    - 27. Cycle number. We adopted cycle numbers recorded in data of the DAC’s. 
    - 28. Time inversion/duplication flag Ft. Ft = 1 if at least one duplicate or inversion of time is found in the sequence containing last fix from the previous cycle and all fixes from the current cycle. Otherwise, Ft =0.

    File WMO2DAC2type.txt catalogs the float information included into
    YoMaHa'07 dataset:

    1 - Serial YoMaHa'07 number of the float
    2 - WMO float ID
    3 - DAC where float data are stored (described in the file DACs.txt)
    4 - Float type (described in the file float_types.txt)
    """

    # parse with panda
    col_names = [
        "lon_d",
        "lat_d",
        "p_d",
        "t_d",
        "u_d",
        "v_d",
        "eu_d",
        "ev_d",
        "lon_s",
        "lat_s",
        "t_s",
        "u_s",
        "v_s",
        "eu_s",
        "ev_s",
        "lon_lp",
        "lat_lp",
        "t_lp",
        "lon_fc",
        "lat_fc",
        "t_fc",
        "lon_lc",
        "lat_lc",
        "t_lc",
        "s_fix",
        "id",
        "cycle",
        "t_inv",
    ]

    na_col = [
        -999.9999,
        -99.9999,
        -999.9,
        -999.999,
        -999.99,
        -999.99,
        -999.99,
        -999.99,
        -999.99,
        -99.99,
        -999.99,
        -999.99,
        -999.99,
        -999.99,
        -999.99,
        -999.99,
        -99.99,
        -999.99,
        -999.99,
        -99.99,
        -999.99,
        -999.99,
        -99.99,
        -999.99,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]

    # open with pandas
    filename = tmp_path + "yomaha07.dat"
    df = pd.read_csv(
        filename, names=col_names, sep="\s+", header=None, na_values=na_col
    )

    # convert to an Xarray Dataset
    ds = xr.Dataset.from_dataframe(df)
    ds = ds.rename_dims({"index": "obs"})

    for t in ["t_s", "t_d", "t_lp", "t_fc", "t_lc"]:
        ds[t].values = pd.to_datetime(ds[t], origin="2000-01-01 00:00", unit="D").values

    unique_id, rowsize = np.unique(ds["id"], return_counts=True)

    ds["id"] = (["traj"], unique_id)
    ds["rowsize"] = (["traj"], rowsize)

    ds = ds.set_coords(["id", "t_d", "t_s", "t_lp", "t_lc", "t_lp"])
    ds = ds.drop_vars("index")

    return ds
