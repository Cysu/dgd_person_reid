#!/usr/bin/env bash
# Extract features. The batch size is assumed to be 100.
# Use GPU 0 by default.

# Set relative paths.
CAFFE_DIR=external/caffe
DB_DIR=external/exp/db
RESULTS_DIR=external/exp/results
MODEL_TEMPLATE=models/exfeat_template.prototxt

parse_args() {
  task=$1
  dataset=$2
  weights=$3
  if [[ $# -eq 4 ]]; then
    blob=$4
  else
    blob=fc7_bn
  fi
}

get_output_dir() {
  parse_args $@
  weights_name=$(basename $weights)
  weights_name="${weights_name%%.*}"
  output_dir=${RESULTS_DIR}/${task}/${dataset}_${weights_name}_${blob}
  echo ${output_dir}
}

extract_features() {
  parse_args $@
  output_dir=$(get_output_dir $@)

  # Extract train, val, test probe, and test gallery features.
  rm -rf ${output_dir}
  mkdir -p ${output_dir}

  for subset in train val test_probe test_gallery; do
    echo "Extracting ${subset} set"
    num_samples=$(wc -l ${DB_DIR}/${dataset}/${subset}.txt | awk '{print $1}')
    num_samples=$((num_samples + 1))
    num_iters=$(((num_samples + 99) / 100))
    model=$(mktemp)
    sed -e "s/\${dataset}/${dataset}/g; s/\${subset}/${subset}/g" \
      ${MODEL_TEMPLATE} > ${model}
    ${CAFFE_DIR}/build/tools/extract_features \
      ${weights} ${model} ${blob},label \
      ${output_dir}/${subset}_features_lmdb,${output_dir}/${subset}_labels_lmdb \
      ${num_iters} lmdb GPU 0
    python2 tools/convert_lmdb_to_numpy.py \
      ${output_dir}/${subset}_features_lmdb ${output_dir}/${subset}_features.npy \
      --truncate ${num_samples}
    python2 tools/convert_lmdb_to_numpy.py \
      ${output_dir}/${subset}_labels_lmdb ${output_dir}/${subset}_labels.npy \
      --truncate ${num_samples}
  done
}