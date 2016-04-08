#!/usr/bin/env bash

RAW=external/raw
EXP=external/exp
CAFFE=external/caffe

cd $(dirname ${BASH_SOURCE[0]})/../

make_db () {
  ROOT_DIR=$1
  DB_DIR=$2
  if [[ $# -eq 4 ]]; then
      RESIZE_HEIGHT=$3
      RESIZE_WIDTH=$4
  else
      RESIZE_HEIGHT=160
      RESIZE_WIDTH=64
  fi

  for subset in train val test_probe test_gallery; do
    if [[ ! -f ${DB_DIR}/${subset}.txt ]]; then
        continue
    fi
    echo "Making ${subset} set"
    $CAFFE/build/tools/convert_imageset \
        ${ROOT_DIR}/ ${DB_DIR}/${subset}.txt ${DB_DIR}/${subset}_lmdb \
        -resize_height ${RESIZE_HEIGHT} -resize_width ${RESIZE_WIDTH}
  done

  echo "Computing images mean"
  $CAFFE/build/tools/compute_image_mean \
      ${DB_DIR}/train_lmdb ${DB_DIR}/mean.binaryproto
}

for d in cuhk03 cuhk01 prid viper 3dpes ilids shinpuhkan; do
    echo "Making $d"
    python2 tools/make_lists_id_training.py $EXP/datasets/$d $EXP/db/$d
    make_db $EXP/datasets/$d $EXP/db/$d
done
