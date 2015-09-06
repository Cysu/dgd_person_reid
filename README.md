# Person Re-identification

The person re-identification project.

## Installation

Basically, there are three external directories should be linked to our project.

First is the directory of raw datasets, which can be downloaded at [here](http://pan.baidu.com/s/1kTy9dUv) with password 8hjx. Make a soft link to the root directory of all the datasets:

    mkdir external
    ln -sf /path/to/the/root/of/datasets external/raw

Next is a directory for our experiments, where later we will put formatted datasets, databases, models, and some results.

    mkdir /path/to/the/experiment/directory
    ln -sf /path/to/the/experiment/directory external/exp

The last is our [brewed caffe](https://github.com/Cysu/caffe/tree/mydev).

    git clone https://github.com/Cysu/caffe.git /path/to/caffe
    cd /path/to/caffe && git checkout mydev && cd -

    # Compile the caffe properly.
    # Please refer to the caffe's repository for detailed instructions.

    # After compilation, link it to our project.
    ln -sf /path/to/caffe external/caffe

## Prepare data

First we need to format raw datasets into our uniform data format.

    scripts/format_rawdata.sh

Next convert the formatted datasets into serialized databases.

    scripts/make_dbs.sh

## Train net

Some model and solver definitions are provided in `models/`. Use `scripts/train_net.sh dataset_name split_index model_name` to train a model on a dataset, for example,

    mkdir external/exp/models && scripts/train_net.sh cuhk03 0 vggnet_bn

## Extract features

After training a CNN, use `scripts/extract_features.sh dataset_split_name model_name caffemodel_path [blob_name]` to extract features, for example,

    scripts/extract_features.sh cuhk03_split_00 vggnet_bn \
        external/exp/models/cuhk03_split_00_vggnet_bn_iter_120000.caffemodel

## Evaluate by CMC

Use the extracted features to learn a metric and then evaluate by CMC score, for example,

    python2 eval/metric_learning.py \
        external/exp/results/cuhk03_split_00_vggnet_bn_fc7_cuhk03_vggnet_bn_iter_120000

This will print several top-k accuracies, and you may find the top-1 accuracy to be around 68%.

## Datasets

We summarize some commonly used person re-id datasets below. They can be downlaoded at [here](http://pan.baidu.com/s/1kTy9dUv) with password 8hjx.

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