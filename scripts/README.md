# This folder contains useful scripts in regards to clouddrift that can help with different tasks such as debugging.

## How to Use
To utilize any of the scripts below make sure to enable executable permissions via: `chmod +x <file-name`

example:
* `chmod +x scripts/get_clib_so_dependencies`

### Currently available scripts:

#### `get_clib_so_dependencies` Script
The purpose of this script is to retrieve details that can be used for debugging when running into issues loading netCDF/hdf6 data files. The script takes two arguments, `env-name` and `output-file-path`.

