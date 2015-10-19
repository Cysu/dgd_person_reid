#!/usr/bin/env bash

RAW=external/raw
EXP=external/exp
CAFFE=external/caffe

cd $(dirname ${BASH_SOURCE[0]})/../

if [[ $# -ne 3 ]]; then
    echo "Usage: $(basename $0) dataset split model"
    echo "    dataset       Dataset name"
    echo "    split         Split index"
    echo "    model         Model name"
    exit
fi

DATASET=$1
printf -v SPLIT_INDEX "%02d" $2
MODEL=$3

# Replace the split_index with our specified one in the template solver and
# template trainval prototxt.
TRAINVAL=models/${DATASET}_split_${SPLIT_INDEX}_${MODEL}_trainval.prototxt
SOLVER=models/${DATASET}_split_${SPLIT_INDEX}_${MODEL}_solver.prototxt
sed -e "s/\${SPLIT_INDEX}/${SPLIT_INDEX}/g" \
    models/${DATASET}_${MODEL}_trainval.prototxt > ${TRAINVAL}
sed -e "s/\${SPLIT_INDEX}/${SPLIT_INDEX}/g" \
    models/${DATASET}_${MODEL}_solver.prototxt > ${SOLVER}

GLOG_logtostderr=1 mpirun -n 2 $CAFFE/build/tools/caffe train \
    -solver ${SOLVER} -gpu 0,1 \
    2>&1 | tee logs/${DATASET}_split_${SPLIT_INDEX}_${MODEL}.log

rm ${TRAINVAL} ${SOLVER}