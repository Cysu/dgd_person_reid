#!/usr/bin/env bash
# Joint single task learning (JSTL) from scratch.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

# Set relative paths.
CAFFE_DIR=external/caffe
MODELS_DIR=models/leave_viper
LOGS_DIR=logs/leave_viper
SNAPSHOTS_DIR=external/exp/snapshots/leave_viper

# Parse arguments.
if [[ $# -ne 2 ]]; then
    echo "Usage: $(basename $0) split model"
    echo "    split         Split index"
    echo "    model         Model name"
    exit
fi
printf -v split_index "%02d" $1
model=$2

# Make directories.
mkdir -p ${LOGS_DIR}
mkdir -p ${SNAPSHOTS_DIR}

# Replace the split_index with our specified one in the template solver and
# template trainval prototxt.
trainval=${MODELS_DIR}/jstl_split_${split_index}_${model}_trainval.prototxt
solver=${MODELS_DIR}/jstl_split_${split_index}_${model}_solver.prototxt
sed -e "s/\${split_index}/${split_index}/g" \
    ${MODELS_DIR}/jstl_${model}_trainval.prototxt > ${trainval}
sed -e "s/\${split_index}/${split_index}/g" \
    ${MODELS_DIR}/jstl_${model}_solver.prototxt > ${solver}

# Train the net.
GLOG_logtostderr=1 mpirun -n 2 ${CAFFE_DIR}/build/tools/caffe train \
    -solver ${solver} -gpu 0,1 \
    2>&1 | tee ${LOGS_DIR}/jstl_split_${split_index}_${model}.log

# Cleanup.
rm ${trainval} ${solver}
