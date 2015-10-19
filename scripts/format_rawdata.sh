#!/usr/bin/env bash

RAW=external/raw
EXP=external/exp

cd $(dirname ${BASH_SOURCE[0]})/../

echo "Formatting CUHK03 ..."
python2 data/format_cuhk03.py $RAW/cuhk03 $EXP/datasets/cuhk03
echo "Formatting VIPeR ..."
python2 data/format_viper.py $RAW/VIPeR $EXP/datasets/viper
echo "Formatting CUHK01 ..."
python2 data/format_cuhk01.py $RAW/cuhk01 $EXP/datasets/cuhk01
echo "Formatting 3DPeS ..."
python2 data/format_3dpes.py $RAW/3DPeS $EXP/datasets/3dpes