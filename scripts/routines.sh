#!/usr/bin/env bash
# Collection of routine functions

# Some constants
CAFFE_DIR=external/caffe

EXP_DIR=external/exp
DATASETS_DIR=${EXP_DIR}/datasets
DB_DIR=${EXP_DIR}/db
RESULTS_DIR=${EXP_DIR}/results
SNAPSHOTS_DIR=${EXP_DIR}/snapshots

MODELS_DIR=models
LOGS_DIR=logs

get_trained_model() {
  local exp=$1
  local dataset=$2

  local solver=${MODELS_DIR}/${exp}/${dataset}_solver.prototxt
  local max_iter=$(grep 'max_iter' ${solver} | awk '{print $2}')
  local snapshot_prefix=$(grep 'snapshot_prefix' ${solver} | awk -F '"' '{print $2}')
  local model=${snapshot_prefix}_iter_${max_iter}.caffemodel
  echo ${model}
}

get_result_dir() {
  local exp=$1
  local dataset=$2
  local trained_model=$3

  local weights_name=$(basename ${trained_model})
  local weights_name="${weights_name%%.*}"
  local result_dir=${RESULTS_DIR}/${exp}/${dataset}_${weights_name}_${blob}
  echo ${result_dir}
}

train_model() {
  local exp=$1
  local dataset=$2
  local pretrained_model=$3

  local solver=${MODELS_DIR}/${exp}/${dataset}_solver.prototxt
  local log=${LOGS_DIR}/${exp}/${dataset}.log
  local trained_model=$(get_trained_model ${exp} ${dataset})

  # Make directories.
  mkdir -p $(dirname ${log})
  mkdir -p $(dirname ${trained_model})

  # Training
  if [[ $# -eq 2 ]]; then
    GLOG_logtostderr=1 mpirun -n 2 ${CAFFE_DIR}/build/tools/caffe train \
      -solver ${solver} -gpu 0,1 2>&1 | tee ${log}
  else
    GLOG_logtostderr=1 mpirun -n 2 ${CAFFE_DIR}/build/tools/caffe train \
      -solver ${solver} -weights ${pretrained_model} -gpu 0,1 2>&1 | tee ${log}
  fi
}

extract_features() {
  local exp=$1
  local dataset=$2
  local trained_model=$3
  if [[ $# -eq 4 ]]; then
    blob=$4
  else
    blob=fc7_bn
  fi

  local result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  rm -rf ${result_dir}
  mkdir -p ${result_dir}

  # Extract train, val, test probe, and test gallery features.
  for subset in train val test_probe test_gallery; do
    echo "Extracting ${subset} set"
    local num_samples=$(wc -l ${DB_DIR}/${dataset}/${subset}.txt | awk '{print $1}')
    local num_samples=$((num_samples + 1))
    local num_iters=$(((num_samples + 99) / 100))
    local model=$(mktemp)
    sed -e "s/\${dataset}/${dataset}/g; s/\${subset}/${subset}/g" \
      ${MODELS_DIR}/exfeat_template.prototxt > ${model}
    ${CAFFE_DIR}/build/tools/extract_features \
      ${trained_model} ${model} ${blob},label \
      ${result_dir}/${subset}_features_lmdb,${result_dir}/${subset}_labels_lmdb \
      ${num_iters} lmdb GPU 0
    python2 tools/convert_lmdb_to_numpy.py \
      ${result_dir}/${subset}_features_lmdb ${result_dir}/${subset}_features.npy \
      --truncate ${num_samples}
    python2 tools/convert_lmdb_to_numpy.py \
      ${result_dir}/${subset}_labels_lmdb ${result_dir}/${subset}_labels.npy \
      --truncate ${num_samples}
  done
}

test_model() {
  local exp=$1
  local dataset=$2
  local trained_model=$3

  # Extract features
  extract_features ${exp} ${dataset} ${trained_model}

  # Evaluate performance
  local result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  python2 eval/metric_learning.py ${result_dir}
}
