#!/bin/bash

REMOTE_GDP_PATH=/phod/pub/lumpkin/netcdf
LOCAL_GDP_PATH=raw/gdp-6hourly

# declare folders list
declare -a folders=("buoydata_1_5000" 
		"buoydata_5001_10000"
		"buoydata_10001_15000" 
		"buoydata_15001_dec20"
        )

# synchronize each of the folders
for folder in "${folders[@]}"
do
   lftp ftp.aoml.noaa.gov -e "mirror -e --ignore-time $REMOTE_GDP_PATH/$folder $LOCAL_GDP_PATH/$folder; quit"
done
