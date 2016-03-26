#!/usr/bin/env bash
# Train a model and test its performance

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

# Directory consts
CAFFE_DIR=external/caffe
MODELS_DIR=models
LOGS_DIR=logs

# Parse arguments.
if [[ $# -ne 2 ]] && [[ $# -ne 3 ]]; then
  echo "Usage: $(basename $0) task dataset [weights]"
  echo "    task          Task name (individually, jstl, jstl_dgd, etc.)"
  echo "    dataset       Dataset name (3dpes, cuhk01, cuhk03, etc.)"
  echo "    weights       Path to pretrained model weights (Optional)"
  exit
fi

task=$1
dataset=$2
weights=$3

# Get related file paths.
get_trained_model() {
  max_iter=$(grep 'max_iter' $1 | awk '{print $2}')
  snapshot_prefix=$(grep 'snapshot_prefix' $1 | awk -F '"' '{print $2}')
  local model=${snapshot_prefix}_iter_${max_iter}.caffemodel
  echo ${model}
}

solver=${MODELS_DIR}/${task}/${dataset}_solver.prototxt
log=${LOGS_DIR}/${task}/${dataset}.log
model=$(get_trained_model ${solver})

# Make directories.
mkdir -p $(dirname ${log})
mkdir -p $(dirname ${model})

# Training
if [[ $# -eq 2 ]]; then
  GLOG_logtostderr=1 mpirun -n 2 ${CAFFE_DIR}/build/tools/caffe train \
    -solver ${solver} -gpu 0,1 2>&1 | tee ${log}
else
  GLOG_logtostderr=1 mpirun -n 2 ${CAFFE_DIR}/build/tools/caffe train \
    -solver ${solver} -weights ${weights} -gpu 0,1 2>&1 | tee ${log}
fi

# Test
source scripts/extract_features.sh
extract_features ${task} ${dataset} ${model}
python2 eval/metric_learning.py ${output_dir}
