#!/usr/bin/env bash

RAW=external/raw
EXP=external/exp
CAFFE=external/caffe

cd $(dirname ${BASH_SOURCE[0]})/../

if [[ $# -ne 2 ]]; then
    echo "Usage: $(basename $0) dataset model"
    echo "    dataset       Dataset name"
    echo "    model         Model name"
    exit
fi

DATASET=$1
MODEL=$2

GLOG_logtostderr=1 mpirun -n 2 $CAFFE/build/tools/caffe train \
    -solver models/${DATASET}_${MODEL}_solver.prototxt -gpu 0,1 \
    | tee models/${DATASET}_${MODEL}.log