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

## Download datasets

Download the following datasets.

1.  [CUHK03](https://docs.google.com/spreadsheet/viewform?usp=drive_web&formkey=dHRkMkFVSUFvbTJIRkRDLWRwZWpONnc6MA#gid=0)
2.  [CUHK01](https://docs.google.com/spreadsheet/viewform?formkey=dF9pZ1BFZkNiMG1oZUdtTjZPalR0MGc6MA)
3.  [PRID](https://lrs.icg.tugraz.at/datasets/prid/prid_2011.zip)
4.  [VIPeR](http://soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip)
5.  [3DPeS](http://imagelab.ing.unimore.it/3DPeS/3dPES_data/3DPeS_ReId_Snap.zip)
6.  [i-LIDS](https://drive.google.com/open?id=0B67_d0rLRTQYRjQ2T3o1NmxvVE0) (I cannot find the link to the original dataset. This is my previous backup version.)
7.  [Shinpuhkan](http://www.mm.media.kyoto-u.ac.jp/en/datasets/shinpuhkan) (need to send an email to the authors)

Link the root directory of these datasets to our project.

    ln -sf /path/to/the/root/of/datasets external/raw

## Prepare data

1.  Create a directory for experiment data and results

        mkdir -p external/exp

    or link against another external directory

        ln -s /path/to/your/exp/directory external/exp

2.  Convert raw datasets into a uniform data format

        scripts/format_rawdata.sh

3.  Convert formatted datasets into LMDBs.

        scripts/make_dbs.sh

4.  Merge all the datasets together for the joint single-task learning (JSTL).

        scripts/merge_dbs.sh

## Experiments

Our experiments are organized into several groups:

1.  Baseline: training individually on each dataset
2.  Baseline: joint single task learning (JSTL)
3.  Our method: domain guided dropout (DGD)
4.  Additional leave-one-out experiments

### Baseline: training individually on each dataset

To train and test a model individually on a dataset, just run the script

    scripts/exp_individually.sh prid

where the parameter is the dataset name, can be one of `cuhk03`, `cuhk01`, `prid`, `viper`, `3dpes`, `ilids`.

### Baseline: joint single task learning (JSTL)

First, pretrain a model using the mixed dataset with JSTL

    scripts/exp_jstl.sh

After training, the script will use the pretrained model to do the evaluation directly on each individual dataset. The CMC accuracies printed out are corresponding to the JSTL entries in Table 3 of our paper.

### Our method: domain guided dropout (DGD)

Based on the pretrained JSTL model, we first compute the neuron impact scores (NIS) for each dataset, and then resume the JSTL training with deterministic DGD.

    scripts/exp_dgd.sh

The CMC accuracies printed out are corresponding to the JSTL+DGD entries in Table 3 of our paper.

At last, to achieve the best performance, we can fine-tune the model on each dataset with stochastic DGD

    scripts/exp_ft_dgd.sh

The CMC accuracies printed out are corresponding to the FT-(JSTL+DGD) entries in Table 3 of our paper.

## Referenced Datasets

We summarize the person re-id datasets used in this project as below.

| Name       | Reference                                                                                                        |
|------------|------------------------------------------------------------------------------------------------------------------|
| 3DPeS      | Baltieri, et al., 3DPes: 3D people dataset for surveillance and forensics                                        |
| CUHK01     | Li, et al., Human reidentification with transferred metric learning                                              |
| CUHK02     | Li, et al., Locally Aligned Feature Transforms across Views                                                      |
| CUHK03     | Li, et al., Deepreid: Deep filter pairing neural network for person re-identification                            |
| i-LIDS     | Zheng, et al., Associating groups of people                                                                      |
| PRID       | Hirzer, et al., Person re-identification by descriptive and discriminative classification                        |
| Shinpuhkan | Kawanishi, et al., Shinpuhkan2014: A Multi-Camera Pedestrian Dataset for Tracking People across Multiple Cameras |
| VIPeR      | Gray, et al., Evaluating appearance models for recognition, reacquisition, and tracking                          |