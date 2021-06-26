#!/usr/bin/env sh

DATAFILE="data/testing.tif"
WEIGHTS="data/storm.pt"
MEANSTD="data/storm_meanstd.mat"
SAVENAME="results.mat"
UPSAMPLING_FACTOR="8"
DEBUG="0"

python Testing.py --datafile $DATAFILE --weights_name $WEIGHTS --meanstd_name $MEANSTD --savename $SAVENAME --upsampling_factor=$UPSAMPLING_FACTOR --debug $DEBUG
