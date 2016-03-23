#!/usr/bin/env bash

RAW=external/raw
EXP=external/exp
CAFFE=external/caffe
DATASETS=$EXP/datasets
DB=$EXP/db

cd $(dirname ${BASH_SOURCE[0]})/../

python2 tools/merge_lists_single_task.py \
    --dataset-dirs $DATASETS/3dpes $DATASETS/cuhk01 $DATASETS/cuhk03 \
                   $DATASETS/ilids $DATASETS/prid $DATASETS/viper \
                   $DATASETS/shinpuhkan \
    --db-dirs $DB/3dpes $DB/cuhk01 $DB/cuhk03 \
              $DB/ilids $DB/prid $DB/viper \
              $DB/shinpuhkan \
    -- $DB/jstl

DB=$DB/jstl
echo "Making training set"
$CAFFE/build/tools/convert_imageset \
    $(pwd)/ $DB/train.txt $DB/train_lmdb \
    -resize_height 160 -resize_width 64

echo "Making validation set"
$CAFFE/build/tools/convert_imageset \
    $(pwd)/ $DB/val.txt $DB/val_lmdb \
    -resize_height 160 -resize_width 64

echo "Computing images mean"
$CAFFE/build/tools/compute_image_mean $DB/train_lmdb $DB/mean.binaryproto
