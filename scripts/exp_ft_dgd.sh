#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='dgd'

# Make a model for inference (treat BN as fixed affine layer)
# to fast the neuron impact scores computation
pretrained_model=$(get_trained_model ${exp} jstl)
python2 ${CAFFE_DIR}/python/gen_bn_inference.py \
  models/jstl/jstl_deploy.prototxt ${pretrained_model}
inference_model=$(get_trained_model_for_inference jstl jstl)

# Compute neuron impact scores (NIS) for each dataset
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  compute_neuron_impact_scores ${dataset} ${inference_model}
done

# Fine-tune on each dataset
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  train_model ${exp} ${dataset} ${pretrained_model}
done

# Extract features on all datasets
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  trained_model=$(get_trained_model ${exp} ${dataset})
  extract_features ${exp} ${dataset} ${trained_model}
done

# Evaluate performance
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  trained_model=$(get_trained_model ${exp} ${dataset})
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  echo ${dataset}
  python2 eval/metric_learning.py ${result_dir}
  echo
done

