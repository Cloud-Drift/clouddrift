from setuptools import setup

setup(
    name="clouddrift",
    version="0.1",
    description="Accelerating the use of Lagrangian data for atmospheric, oceanic, and climate sciences",
    url="https://github.com/Cloud-Drift/clouddrift",
    author="Philippe Miron",
    author_email="pmiron@fsu.edu",
    license="Apache-2.0 license",
    packages=["clouddrift"],
    zip_safe=False,
    install_requires=[
        "cartopy==0.20.2",
        "matplotlib==3.5.2",
        "numpy==1.22.4",
        "pandas==1.4.2",
        "xarray==2022.3.0",
        "pyarrow==8.0.0",
        "zarr==2.11.3",
        "numba==0.53.1",
        "tqdm==4.64.0",
        "awkward@https://github.com/scikit-hep/awkward.git@1.9.0rc4",
    ],
)
