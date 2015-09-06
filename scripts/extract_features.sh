#!/usr/bin/env bash
# Extract features. The batch size is assumed to be 100.

RAW=external/raw
EXP=external/exp
CAFFE=external/caffe

if [[ $# -ne 3 ]] && [[ $# -ne 4 ]]; then
  echo "Usage: $(basename $0) dataset model weights [blob=fc7]"
  echo "    dataset         Dataset name where features are extracted"
  echo "    model           Model name"
  echo "    weights         Pretrained caffe model weights"
  echo "    blob            Name of the blob to be extracted. Default fc7"
  exit
fi

DATASET=$1
MODEL=$2
WEIGHTS=$3
if [[ $# -eq 4 ]]; then
  BLOB=$4
else
  BLOB=fc7
fi

TEMPLATE=models/exfeat_template_${MODEL}.prototxt
WEIGHTS_NAME=$(basename $WEIGHTS)
WEIGHTS_NAME="${WEIGHTS_NAME%%.*}"
OUTPUT=${EXP}/results/${DATASET}_${MODEL}_${BLOB}_${WEIGHTS_NAME}

mkdir -p ${OUTPUT}
for token in train val test_probe test_gallery; do
  echo "Extracting ${token} set"
  num_samples=$(wc -l ${EXP}/db/${DATASET}/${token}.txt | awk '{print $1}')
  num_iters=$((num_samples / 100 + 1))
  tmp_model=$(mktemp)
  sed -e "s/\${DATASET}/${DATASET}/g; s/\${TOKEN}/${token}/g" \
      ${TEMPLATE} > ${tmp_model}
  ${CAFFE}/build/tools/extract_features \
      ${WEIGHTS} ${tmp_model} ${BLOB} ${OUTPUT}/${token}_features_lmdb \
      ${num_iters} lmdb GPU 0
  ${CAFFE}/build/tools/extract_features \
      ${WEIGHTS} ${tmp_model} label ${OUTPUT}/${token}_labels_lmdb \
      ${num_iters} lmdb GPU 0
  python2 tools/convert_lmdb_to_numpy.py \
      ${OUTPUT}/${token}_features_lmdb ${OUTPUT}/${token}_features.npy \
      --truncate ${num_samples}
  python2 tools/convert_lmdb_to_numpy.py \
      ${OUTPUT}/${token}_labels_lmdb ${OUTPUT}/${token}_labels.npy \
      --truncate ${num_samples}
done