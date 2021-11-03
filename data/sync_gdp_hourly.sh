#!/bin/bash

REMOTE_GDP_PATH=/phod/pub/lumpkin/hourly/v1.04/netcdf
LOCAL_GDP_PATH=raw


# declare folders list
declare -a folders=("argos_block1" 
		    "argos_block2"
		    "argos_block3" 
		    "argos_block4"
	    	"argos_block5"
		    "argos_block6"
		    "argos_block7"
		    "argos_block8"
			"gps")

# synchronize each of the folders
for folder in "${folders[@]}"
do
   lftp ftp.aoml.noaa.gov -e "mirror -e --ignore-time $REMOTE_GDP_PATH/$folder $LOCAL_GDP_PATH/$folder; quit"
done
