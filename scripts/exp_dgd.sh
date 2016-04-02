#!/usr/bin/env bash
# Experiments of joint single task learning (JSTL) with domain guided dropout (DGD)

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='dgd'

Make a model for inference (treat BN as fixed affine layer)
to fast the neuron impact scores computation
trained_model=$(get_trained_model jstl jstl)
python2 ${CAFFE_DIR}/python/gen_bn_inference.py \
  models/jstl/jstl_deploy.prototxt ${trained_model}
inference_model=$(get_trained_model_for_inference jstl jstl)

# Compute neuron impact scores (NIS) for each dataset
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids shinpuhkan; do
  compute_neuron_impact_scores ${dataset} ${inference_model}
done

# Combine the NIS together for JSTL data samples
for subset in train val; do
  python2 tools/save_joint_impact_score.py \
    ${NIS_DIR} ${DB_DIR}/jstl/${subset}.txt ${DB_DIR}/jstl/nis_${subset}_lmdb
done

# Resume the JSTL training with DGD
train_model ${exp} jstl ${trained_model}
trained_model=$(get_trained_model ${exp} jstl)

# Extract features on all datasets
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  extract_features ${exp} ${dataset} ${trained_model}
done

# Evaluate performance
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  echo ${dataset}
  python2 eval/metric_learning.py ${result_dir}
  echo
done
