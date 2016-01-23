#!/usr/bin/env bash
# Compute the neurons impact score for each dataset.
# The batch size is assumed to be 20.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

# Set relative paths.
MODELS_DIR=models/fc_only
DB_DIR=external/exp/db
SNAPSHOTS_DIR=external/exp/snapshots/fc_only

# Parse arguments.
if [[ $# -ne 5 ]] && [[ $# -ne 6 ]]; then
    echo "Usage: $(basename $0) dataset split model weights output_dir [layer=fc7]"
    echo "    dataset       Dataset name"
    echo "    split         Split index"
    echo "    model         Model name"
    echo "    weights       Pretrained caffe model weights"
    echo "    output_dir    Output directory"
    echo "    layer         Name of the layer. Default fc7."
    exit
fi
dataset=$1
printf -v split_index "%02d" $2
model=$3
weights=$4
output_dir=$5
if [[ $# -eq 6 ]]; then
  layer=$6
else
  layer=fc7
fi

# Make directories.
mkdir -p ${output_dir}

# Replace the split_index with our specified one in the template solver and
# template trainval prototxt.
trainval=${MODELS_DIR}/${dataset}_split_${split_index}_${model}_trainval.prototxt
sed -e "s/\${split_index}/${split_index}/g" \
    ${MODELS_DIR}/${dataset}_${model}_trainval.prototxt > ${trainval}

# Get number of samples in the validation set.
dbname=${dataset}_split_${split_index}
num_samples=$(wc -l ${DB_DIR}/${dbname}/val.txt | awk '{print $1}')
num_samples=$((num_samples + 1))
num_iters=$((num_samples / 20 + 1))

# Get the output name.
weights_name=$(basename $weights)
weights_name="${weights_name%%.*}"
output_npy=${output_dir}/${dataset}.npy

# Compute.
python2 tools/compute_impact_score.py ${trainval} ${weights} ${output_npy} \
    --num_iters ${num_iters} --layer ${layer} --normalize

# Cleanup.
rm ${trainval}
