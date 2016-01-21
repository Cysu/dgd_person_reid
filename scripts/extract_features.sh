#!/usr/bin/env bash
# Extract features. The batch size is assumed to be 100.
# Use GPU 0 by default.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

# Set relative paths.
CAFFE_DIR=external/caffe
DB_DIR=external/exp/db
RESULTS_DIR=external/exp/results

# Parse arguments.
if [[ $# -ne 3 ]] && [[ $# -ne 4 ]]; then
  echo "Usage: $(basename $0) db model weights [blob=fc7_bn]"
  echo "    dbname          Database name where features are extracted"
  echo "    model           Model name"
  echo "    weights         Pretrained caffe model weights"
  echo "    blob            Name of the blob to be extracted. Default fc7_bn."
  exit
fi
dbname=$1
model=$2
weights=$3
if [[ $# -eq 4 ]]; then
  blob=$4
else
  blob=fc7_bn
fi

# Get file names for prototxt, caffemodel, and output directory.
model_template=models/exfeat_template_${model}.prototxt
weights_name=$(basename $weights)
weights_name="${weights_name%%.*}"
output_dir=${RESULTS_DIR}/${dbname}_${model}_${blob}_${weights_name}

# Extract train, val, test probe, and test gallery features.
rm -rf ${output_dir}
mkdir -p ${output_dir}
for subset in train val test_probe test_gallery; do
  echo "Extracting ${subset} set"
  num_samples=$(wc -l ${DB_DIR}/${dbname}/${subset}.txt | awk '{print $1}')
  num_samples=$((num_samples + 1))
  num_iters=$((num_samples / 100 + 1))
  model=$(mktemp)
  sed -e "s/\${dbname}/${dbname}/g; s/\${subset}/${subset}/g" \
      ${model_template} > ${model}
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
