#!/usr/bin/env bash

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

# Set relative paths.
DB_DIR=external/exp/db

# Parse arguments.
if [[ $# -ne 3 ]]; then
    echo "Usage: $(basename $0) split impact_dir joint_set"
    echo "    split         Split index"
    echo "    impact_dir    A directory of impact scores numpy files"
    echo "    joint_set     Joint dataset name, e.g., jstl, leave_cuhk01"
    exit
fi
printf -v split_index "%02d" $1
impact_dir=$2
prefix=$(basename ${impact_dir})
joint_set=$3

# Save individual
for ds in $(ls ${impact_dir}); do
  ds=$(basename ${ds} .npy)
  python2 tools/save_individual_impact_score.py \
      ${impact_dir}/${ds}.npy \
      ${DB_DIR}/${ds}_split_${split_index}/${prefix}_lmdb
  echo "Saved for ${ds}"
done

# Save joint
python2 tools/save_joint_impact_score.py \
    ${impact_dir} \
    ${DB_DIR}/${joint_set}_split_${split_index}/train.txt \
    ${DB_DIR}/${joint_set}_split_${split_index}/${prefix}_train_lmdb
echo "Saved for ${joint_set} train"
python2 tools/save_joint_impact_score.py \
    ${impact_dir} \
    ${DB_DIR}/${joint_set}_split_${split_index}/val.txt \
    ${DB_DIR}/${joint_set}_split_${split_index}/${prefix}_val_lmdb
echo "Saved for ${joint_set} val"