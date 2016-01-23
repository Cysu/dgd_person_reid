#!/usr/bin/env bash

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

# Set relative paths.
DB_DIR=external/exp/db

# Parse arguments.
if [[ $# -ne 2 ]]; then
    echo "Usage: $(basename $0) split impact_dir"
    echo "    split         Split index"
    echo "    impact_dir    A directory of impact scores numpy files"
    exit
fi
printf -v split_index "%02d" $1
impact_dir=$2
prefix=$(basename ${impact_dir})

# Save individual
for ds in cuhk03 cuhk01 prid viper 3dpes ilids shinpuhkan; do
  python2 tools/save_individual_impact_score.py \
      ${impact_dir}/${ds}.npy \
      ${DB_DIR}/${ds}_split_${split_index}/${prefix}_lmdb
  echo "Saved for ${ds}"
done

# Save joint
python2 tools/save_joint_impact_score.py \
    ${impact_dir} \
    ${DB_DIR}/jstl_split_${split_index}/train.txt \
    ${DB_DIR}/jstl_split_${split_index}/${prefix}_train_lmdb
echo "Saved for jstl train"
python2 tools/save_joint_impact_score.py \
    ${impact_dir} \
    ${DB_DIR}/jstl_split_${split_index}/val.txt \
    ${DB_DIR}/jstl_split_${split_index}/${prefix}_val_lmdb
echo "Saved for jstl val"