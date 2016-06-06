#!/usr/bin/env bash
# Extract features. The batch size is assumed to be 100.
# Use GPU 0 by default.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

if [[ $# -ne 3 ]] && [[ $# -ne 4 ]]; then
  echo "Usage: $(basename $0) exp dataset weights [blob=fc7_bn]"
  echo "    exp      Subfolder name to store the extracted features"
  echo "    dataset  Dataset name (3dpes, cuhk01, cuhk03, ilids, prid, viper)"
  echo "    weights  Trained caffemodel"
  echo "    blob     Features blob name (fc7_bn by default)"
  exit
fi

extract_features "$@"
