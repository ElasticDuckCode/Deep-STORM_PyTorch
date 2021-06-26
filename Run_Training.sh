#!/usr/bin/env sh

FILENAME="data/TrainingSet.mat"
WEIGHTS="data/storm.pt"
MEANSTD="data/storm_meanstd.mat"

python Training.py --filename $FILENAME --weights_name $WEIGHTS --meanstd_name $MEANSTD
