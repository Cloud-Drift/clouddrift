#!/bin/bash

REMOTE_GDP_PATH=/phod/pub/lumpkin/hourly/v2.00/netcdf
LOCAL_GDP_PATH=raw/gdp-v2.00

lftp ftp.aoml.noaa.gov -e "mirror -e --ignore-time $REMOTE_GDP_PATH $LOCAL_GDP_PATH; quit"
