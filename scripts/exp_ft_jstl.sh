#!/usr/bin/env bash
# Fine-tune on a particular dataset with standard dropout from JSTL.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

# Set relative paths.
CAFFE_DIR=external/caffe
MODELS_DIR=models/ft_jstl
LOGS_DIR=logs/ft_jstl
SNAPSHOTS_DIR=external/exp/snapshots/ft_jstl

# Parse arguments.
if [[ $# -ne 4 ]]; then
    echo "Usage: $(basename $0) dataset split model weights"
    echo "    dataset       Dataset name"
    echo "    split         Split index"
    echo "    model         Model name"
    echo "    weights       Pretrained caffe model weights"
    exit
fi
dataset=$1
printf -v split_index "%02d" $2
model=$3
weights=$4

# Make directories.
mkdir -p ${LOGS_DIR}
mkdir -p ${SNAPSHOTS_DIR}

# Replace the split_index with our specified one in the template solver and
# template trainval prototxt.
TRAINVAL=${MODELS_DIR}/${dataset}_split_${split_index}_${model}_trainval.prototxt
SOLVER=${MODELS_DIR}/${dataset}_split_${split_index}_${model}_solver.prototxt
sed -e "s/\${split_index}/${split_index}/g" \
    ${MODELS_DIR}/${dataset}_${model}_trainval.prototxt > ${TRAINVAL}
sed -e "s/\${split_index}/${split_index}/g" \
    ${MODELS_DIR}/${dataset}_${model}_solver.prototxt > ${SOLVER}

# Fine-tuning.
GLOG_logtostderr=1 mpirun -n 2 ${CAFFE_DIR}/build/tools/caffe train \
    -solver ${SOLVER} -weights ${weights} -gpu 0,1 \
    2>&1 | tee ${LOGS_DIR}/${dataset}_split_${split_index}_${model}.log

# Cleanup.
rm ${TRAINVAL} ${SOLVER}
