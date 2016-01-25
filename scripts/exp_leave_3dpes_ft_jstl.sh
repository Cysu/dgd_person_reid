#!/usr/bin/env bash
# Fine-tune on a particular dataset with standard dropout from JSTL.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

# Set relative paths.
CAFFE_DIR=external/caffe
MODELS_DIR=models/leave_3dpes
LOGS_DIR=logs/leave_3dpes
SNAPSHOTS_DIR=external/exp/snapshots/leave_3dpes

# Parse arguments.
if [[ $# -ne 3 ]]; then
    echo "Usage: $(basename $0) split model weights"
    echo "    split         Split index"
    echo "    model         Model name"
    echo "    weights       Pretrained caffe model weights"
    exit
fi
printf -v split_index "%02d" $1
model=$2
weights=$3

# Make directories.
mkdir -p ${LOGS_DIR}
mkdir -p ${SNAPSHOTS_DIR}

# Replace the split_index with our specified one in the template solver and
# template trainval prototxt.
trainval=${MODELS_DIR}/ft_jstl_split_${split_index}_${model}_trainval.prototxt
solver=${MODELS_DIR}/ft_jstl_split_${split_index}_${model}_solver.prototxt
sed -e "s/\${split_index}/${split_index}/g" \
    ${MODELS_DIR}/ft_jstl_${model}_trainval.prototxt > ${trainval}
sed -e "s/\${split_index}/${split_index}/g" \
    ${MODELS_DIR}/ft_jstl_${model}_solver.prototxt > ${solver}

# Fine-tuning.
GLOG_logtostderr=1 mpirun -n 2 ${CAFFE_DIR}/build/tools/caffe train \
    -solver ${solver} -weights ${weights} -gpu 0,1 \
    2>&1 | tee ${LOGS_DIR}/ft_jstl_split_${split_index}_${model}.log

# Cleanup.
rm ${trainval} ${solver}
