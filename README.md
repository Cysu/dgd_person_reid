# Domain Guided Dropout for Person Re-id

This project aims at learning generic person re-identification (re-id) deep features from multiple datasets with domain guided dropout. Mainly based on our CVPR 2016 paper [Learning Deep Feature Representations with Domain Guided Dropout for Person Re-identification](Learning Deep Feature Representations with Domain Guided Dropout for Person Re-identification).

## Installation

We have integrated our self-brewed caffe into `external/caffe`, which provides batch-normalization and multi-gpu parallel training. Please clone this project with the command:

    git clone --recursive https://github.com/Cysu/dgd_person_reid.git

Apart from the official installation [prerequisites](http://caffe.berkeleyvision.org/installation.html), we have several other dependencies: cudnn-v4, openmpi, and 1.55 <= boost < 1.60. You may install them manually or by a package manager (a tip for installing boost 1.55 on Ubuntu 14.04: `sudo apt-get autoremove libboost1.54*` then `sudo apt-get install libboost1.55-all-dev`).

Then configure the `Makefile.config` and compile the caffe. To use multi-GPU for training, please uncomment the MPI parallel block in the `Makefile.config` and set the `MPI_INCLUDE` and `MPI_LIB` properly. Please find more details of using the caffe [here](https://github.com/Cysu/caffe/tree/domain-guided-dropout).

    cd external/caffe
    cp Makefile.config.example Makefile.config
    # Configure the libraries properly
    make -j8 && make py

Some other prerequisites are

1.  Matlab (to pre-process the CUHK03 dataset)
2.  python2 packages: numpy, scipy, Pillow, scikit-learn, protobuf, lmdb
3.  Add `export PYTHONPATH=".:$PYTHONPATH"` to `~/.bashrc` and restart the terminal

## Download datasets

Download the following datasets.

1.  [CUHK03](https://docs.google.com/spreadsheet/viewform?usp=drive_web&formkey=dHRkMkFVSUFvbTJIRkRDLWRwZWpONnc6MA#gid=0)
2.  [CUHK01](https://docs.google.com/spreadsheet/viewform?formkey=dF9pZ1BFZkNiMG1oZUdtTjZPalR0MGc6MA)
3.  [PRID](https://lrs.icg.tugraz.at/datasets/prid/prid_2011.zip)
4.  [VIPeR](http://soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip)
5.  [3DPeS](http://imagelab.ing.unimore.it/3DPeS/3dPES_data/3DPeS_ReId_Snap.zip)
6.  [i-LIDS](https://drive.google.com/file/d/0B67_d0rLRTQYRjQ2T3o1NmxvVE0/view?usp=sharing) (I cannot find the link to the original dataset. This is my previous backup version.)
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

3.  Convert formatted datasets into LMDBs

        scripts/make_dbs.sh

4.  Merge all the datasets together for the joint single-task learning (JSTL)

        scripts/merge_dbs.sh

## Experiments

**Note: We use two GPUs to train the models by default. Change the `mpirun -n 2 ... -gpu 0,1` in `scripts/routines.sh` to your own hardware configuration if necessary.**

Our experiments are organized into two groups:

1.  Baseline: training individually on each dataset
2.  Ours: Joint single task learning (JSTL) + Domain guided dropout (DGD)

We provide a **pretrained JSTL+DGD model [here](https://drive.google.com/open?id=0B67_d0rLRTQYZnB5ZUZpdTlxM0k)** that can be used as a generic person re-id feature extractor.

Some archived experiment logs can be found at `archived/`.

### Baseline: training individually on each dataset

To train and test a model individually on a dataset, just run the script

    scripts/exp_individually.sh prid

where the parameter is the dataset name, can be one of `cuhk03`, `cuhk01`, `prid`, `viper`, `3dpes`, `ilids`.

### Ours: Joint single task learning (JSTL) + Domain guided dropout (DGD)

1. Pretrain a model using the mixed dataset with JSTL. The CMC accuracies printed out are corresponding to the **JSTL** entries in Table 3 of our paper.

        scripts/exp_jstl.sh

2. Based on the pretrained JSTL model, we first compute the neuron impact scores (NIS) for each dataset, and then resume the JSTL training with deterministic DGD. The CMC accuracies printed out are corresponding to the **JSTL+DGD** entries in Table 3 of our paper.

        scripts/exp_dgd.sh

    At last, to achieve the best performance, we can fine-tune the model on each dataset with stochastic DGD. The CMC accuracies printed out are corresponding to the **FT-(JSTL+DGD)** entries in Table 3 of our paper.

        scripts/exp_ft_dgd.sh

## Citation

    @inproceedings{xiao2016learning,
      title={Learning Deep Feature Representations with Domain Guided Dropout for Person Re-identification},
      author={Xiao, Tong and Li, Hongsheng and Ouyang, Wanli and Wang, Xiaogang},
      booktitle={CVPR},
      year={2016}
    }

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
