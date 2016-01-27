# Person Re-identification

This is a person re-identification (re-id) project that aims at learning generic deep features from multiple person re-id datasets.

## Installation

We have integrated our self-brewed caffe into `external/caffe`, which provides batch-normalization and multi-gpu parallel training. Please clone this project with the command:

    git clone --recursive https://github.com/Cysu/person_reid.git

Apart from the official installation [prerequisites](http://caffe.berkeleyvision.org/installation.html), we have several other dependencies: [cudnn-v4](https://developer.nvidia.com/cudnn), openmpi, and 1.55 <= boost < 1.60. You may install them manually or by a package manager (apt-get, pacman, yum, etc.).

Then configure the `Makefile.config` and compile the caffe.

    cd external/caffe
    cp Makefile.config.example Makefile.config
    # Configure the libraries properly
    make -j8 && make py

## Setup environment for experiments

First, download some person re-id datasets at [here](http://pan.baidu.com/s/1kTy9dUv) with password 8hjx. Make a soft link to the root directory:

    ln -sf /path/to/the/root/of/datasets external/raw

Next, create a directory for our experiments, where later we will put formatted datasets, databases, snapshots, and some results.

    mkdir -p /path/to/the/experiment/directory
    ln -sf /path/to/the/experiment/directory external/exp

## Prepare data

First we need to format raw datasets into our uniform data format.

    scripts/format_rawdata.sh

Next convert each formatted dataset into a LMDB.

    scripts/make_dbs.sh

At last merge all the datasets together for the joint single-task learning (JSTL).

    scripts/merge_dbs.sh

## Train nets

Some model and solver definitions are listed in `models/`. Use the shell scripts provided in `scripts/` to run the experiments. **Note that by default the scripts will use two GPUs. You may find the command `mpirun -n 2 ... -gpu 0,1` in the scripts and adapt it with your own settings.**

As an example, training a net by Joint Single-Task Learning (JSTL) is done by

    scripts/exp_jstl.sh 0 googlenet_bn

The snapshots will be saved in `external/exp/snapshots/jstl/`.

## Evaluate the performance

After training, use `scripts/extract_features.sh dataset_split_name model_name caffemodel_path [blob_name]` to extract features, for example,

    scripts/extract_features.sh cuhk03_split_00 googlenet_bn \
      external/exp/snapshots/jstl/jstl_split_00_googlenet_bn_iter_55000.caffemodel \
      fc7_bn

Then learn a metric and evaluate by CMC score, for example,

    python2 eval/metric_learning.py \
      external/exp/results/cuhk03_split_00_googlenet_bn_fc7_bn_jstl_split_00_googlenet_bn_iter_55000

This will print several top-k accuracies, and you may find the top-1 accuracy to be around 73%.

## Datasets

We summarize some commonly used person re-id datasets below. They can be downlaoded from [here](http://pan.baidu.com/s/1kTy9dUv) with password 8hjx.

<table>
  <tr>
    <th>Name</th>
    <th>Reference</th>
  </tr>
  <tr>
    <td>3DPeS</td>
    <td>D. Baltieri, et al., 3DPes: 3D people dataset for surveillance and forensics</td>
  </tr>
  <tr>
    <td>CUHK01</td>
    <td>W. Li, et al., Human reidentification with transferred metric learning</td>
  </tr>
  <tr>
    <td>CUHK02</td>
    <td>W. Li, et al., Locally Aligned Feature Transforms across Views</td>
  </tr>
  <tr>
    <td>CUHK03</td>
    <td>W. Li, et al., Deepreid: Deep filter pairing neural network for person re-identification</td>
  </tr>
  <tr>
    <td>i-LIDS</td>
    <td>W. Zheng, et al., Associating groups of people</td>
  </tr>
  <tr>
    <td>i-LIDS-VID</td>
    <td>T. Wang, et al., Person Re-Identification by Video Ranking</td>
  </tr>
  <tr>
    <td>Market-1501</td>
    <td>L. Zheng, et al., Scalable Person Re-identification: A Benchmark</td>
  </tr>
  <tr>
    <td>OPeRID</td>
    <td>S. Liao, et al., Open-set Person Re-identification</td>
  </tr>
  <tr>
    <td>PRID</td>
    <td>M. Hirzer, et al., Person re-identification by descriptive and discriminative classification</td>
  </tr>
  <tr>
    <td>RAiD</td>
    <td>A. Das et al., Consistent re-identification in a camera network</td>
  </tr>
  <tr>
    <td>Shinpuhkan</td>
    <td>Y. Kawanishi, et al., Shinpuhkan2014: A Multi-Camera Pedestrian Dataset for Tracking People across Multiple Cameras</td>
  </tr>
  <tr>
    <td>VIPeR</td>
    <td>D. Gray, et al., Evaluating appearance models for recognition, reacquisition, and tracking</td>
  </tr>
</table>