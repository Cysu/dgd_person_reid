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
    echo "Making ${subset} set"
    $CAFFE/build/tools/convert_imageset \
        ${ROOT_DIR}/ ${DB_DIR}/${subset}.txt ${DB_DIR}/${subset}_lmdb \
        -resize_height ${RESIZE_HEIGHT} -resize_width ${RESIZE_WIDTH}
  done

  echo "Computing images mean"
  $CAFFE/build/tools/compute_image_mean \
      ${DB_DIR}/train_lmdb ${DB_DIR}/mean.binaryproto
}


# cuhk03
for i in {00..04}; do
  echo "Making cuhk03 split $i"
  python2 tools/make_lists_id_training.py \
      $EXP/datasets/cuhk03/ $EXP/db/cuhk03_split_$i --split-index $i
  make_db $EXP/datasets/cuhk03 $EXP/db/cuhk03_split_$i
done

# viper
for i in {00..09}; do
  echo "Making viper split $i"
  python2 tools/make_lists_id_training.py \
      $EXP/datasets/viper/ $EXP/db/viper_split_$i --split-index $i
  make_db $EXP/datasets/viper $EXP/db/viper_split_$i
done

# cuhk01
for i in {00..09}; do
  echo "Making cuhk01 split $i"
  python2 tools/make_lists_id_training.py \
      $EXP/datasets/cuhk01/ $EXP/db/cuhk01_split_$i --split-index $i
  make_db $EXP/datasets/cuhk01 $EXP/db/cuhk01_split_$i
done

# 3dpes
for i in {00..09}; do
  echo "Making 3dpes split $i"
  python2 tools/make_lists_id_training.py \
      $EXP/datasets/3dpes/ $EXP/db/3dpes_split_$i --split-index $i
  make_db $EXP/datasets/3dpes $EXP/db/3dpes_split_$i
done

# ilids
for i in {00..09}; do
  echo "Making ilids split $i"
  python2 tools/make_lists_id_training.py \
      $EXP/datasets/ilids/ $EXP/db/ilids_split_$i --split-index $i
  make_db $EXP/datasets/ilids $EXP/db/ilids_split_$i
done

# prid
for i in {00..00}; do
  echo "Making prid split $i"
  python2 tools/make_lists_id_training.py \
      $EXP/datasets/prid/ $EXP/db/prid_split_$i --split-index $i
  make_db $EXP/datasets/prid $EXP/db/prid_split_$i
done