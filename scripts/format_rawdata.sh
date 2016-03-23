#!/usr/bin/env bash

cd $(dirname ${BASH_SOURCE[0]})/../

RAW=external/raw
EXP=external/exp

echo "Formatting CUHK03 ..."
unzip -q -d $RAW/ $RAW/cuhk03_release.zip
# Save the matfile in the v7 format to fast computation
cd $RAW/cuhk03_release
matlab -nodisplay -nojvm -nosplash -r "load('cuhk-03.mat'); save('cuhk-03.mat', 'detected', 'labeled', 'testsets', '-v7'); exit;"
cd -
python2 data/format_cuhk03.py $RAW/cuhk03_release $EXP/datasets/cuhk03

echo "Formatting CUHK01 ..."
unzip -q -d $RAW/cuhk01/ $RAW/CUHK01.zip
python2 data/format_cuhk01.py $RAW/cuhk01/campus $EXP/datasets/cuhk01

echo "Formatting PRID ..."
unzip -q -d $RAW/prid/ $RAW/prid_2011.zip
python2 data/format_prid.py $RAW/prid $EXP/datasets/prid

echo "Formatting VIPeR ..."
unzip -q -d $RAW/ $RAW/VIPeR.v1.0.zip
python2 data/format_viper.py $RAW/VIPeR $EXP/datasets/viper

echo "Formatting 3DPeS ..."
unzip -q -d $RAW/ $RAW/3DPeS_ReId_Snap.zip
python2 data/format_3dpes.py $RAW/3DPeS $EXP/datasets/3dpes

echo "Formatting i-LIDS ..."
tar xf $RAW/i-LIDS.tar.gz -C $RAW/
python2 data/format_ilids.py $RAW/i-LIDS $EXP/datasets/ilids

echo "Formatting Shinpuhkan ..."
unzip -q -d $RAW/ $RAW/Shinpuhkan2014dataset.zip
python2 data/format_shinpuhkan.py $RAW/Shinpuhkan2014dataset $EXP/datasets/shinpuhkan
