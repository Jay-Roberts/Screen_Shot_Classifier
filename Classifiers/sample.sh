#!/bin/bash
cd estimator

# Local files to copy the data to
# gmodel's parser is expecting a directory organized by gameID
FILE_DIR=../TFRecords
DATA_DIR=TFRecords
cp -r $FILE_DIR $DATA_DIR

# Get the data from the bucket
#gsutil cp $GCS_FILES $FILE_DIR

export TRAIN_STEPS=10000
DATE=`date '+%Y%m%d_%H%M%S'`
export OUTPUT_DIR=output_$DATE

#Local training
python trainer/gtask.py --model SNN \
                       --file-dir $DATA_DIR \
                       --job-dir $OUTPUT_DIR \
                       --num-epochs 1\
                       --eval-steps 100 \
                       --train-steps 1000


# Get rid of local copy of data
rm -r $DATA_DIR

