#!/usr/bin/env bash
# Experiments of training and testing a model individually on each dataset

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

# Parse arguments.
if [[ $# -ne 1 ]]; then
  echo "Usage: $(basename $0) dataset"
  echo "    dataset    Dataset name (3dpes, cuhk01, cuhk03, ilids, prid, viper)"
  exit
fi

exp='individually'
dataset=$1

train_model ${exp} ${dataset}
test_model ${exp} ${dataset} $(get_trained_model ${exp} ${dataset})