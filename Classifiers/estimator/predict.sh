#!/bin/bash

# This script picks the most recently trained NN 
# and loads it for inference/predictions on 
# new data sets

CHKPT_FILE=$(find . -type f -name model.ckpt* -exec stat --format '%Y :%y %n' "{}" \; \
                | sort -nr \
                | cut -d: -f2- \
                | head -n 1 \
                | cut -d ' ' --fields 4 \
                | cut -d . --fields 2,3)
echo $CHKPT_FILE

python ./predictor.py -d $CHKPT_FILE
