import numpy as np
from argparse import ArgumentParser
from scipy.misc import imsave

from utils import *


def _load(cuhk03_dir):
    try:
        from scipy.io import loadmat
        matdata = loadmat(osp.join(cuhk03_dir, 'cuhk-03.mat'))
    except:
        from hdf5storage import loadmat
        matdata = loadmat(osp.join(cuhk03_dir, 'cuhk-03.mat'))
    return matdata


def main(args):
    matdata = _load(args.cuhk03_dir)
    output_dir = args.output_dir
    # Although there are 5 pairs of camera views, we tile them up as one pair.
    mkdir_if_missing(osp.join(args.output_dir, 'cam_0'))
    mkdir_if_missing(osp.join(args.output_dir, 'cam_1'))
    identities = []
    for imgs_labeled, imgs_detected in zip(
            matdata['labeled'].squeeze(), matdata['detected'].squeeze()):
        # We merge the manually labeled and automatically detected images of
        # the same view.
        for i in xrange(imgs_labeled.shape[0]):
            pid = len(identities)
            p_images = []
            # view-0
            v_images = []
            for j in xrange(5):
                if imgs_labeled[i, j].size == 0:
                    break
                file_name = 'cam_0/{:05d}_{:05d}.jpg'.format(pid, len(v_images))
                imsave(osp.join(output_dir, file_name), imgs_labeled[i, j])
                v_images.append(file_name)
            for j in xrange(5):
                if imgs_detected[i, j].size == 0:
                    break
                file_name = 'cam_0/{:05d}_{:05d}.jpg'.format(pid, len(v_images))
                imsave(osp.join(output_dir, file_name), imgs_detected[i, j])
                v_images.append(file_name)
            p_images.append(v_images)
            # view-1
            v_images = []
            for j in xrange(5, 10):
                if imgs_labeled[i, j].size == 0:
                    break
                file_name = 'cam_1/{:05d}_{:05d}.jpg'.format(pid, len(v_images))
                imsave(osp.join(output_dir, file_name), imgs_labeled[i, j])
                v_images.append(file_name)
            for j in xrange(5, 10):
                if imgs_detected[i, j].size == 0:
                    break
                file_name = 'cam_1/{:05d}_{:05d}.jpg'.format(pid, len(v_images))
                imsave(osp.join(output_dir, file_name), imgs_detected[i, j])
                v_images.append(file_name)
            p_images.append(v_images)
            identities.append(p_images)
    # Save meta information into a json file
    meta = {'name': 'cuhk03', 'shot': 'multiple', 'num_cameras': 2}
    meta['identities'] = identities
    write_json(meta, osp.join(output_dir, 'meta.json'))
    # Save training and test splits into a json file
    view_counts = [a.shape[0] for a in matdata['labeled'].squeeze()]
    vid_offsets = np.r_[0, np.cumsum(view_counts)]
    test_info = np.random.choice(matdata['testsets'].squeeze())
    test_pids = []
    for i, j in test_info:
        pid = vid_offsets[i - 1] + j - 1
        test_pids.append(pid)
    test_pids.sort()
    trainval_pids = list(set(xrange(vid_offsets[-1])) - set(test_pids))
    split = {'trainval': trainval_pids,
             'test_probe': test_pids,
             'test_gallery': test_pids}
    write_json(split, osp.join(output_dir, 'split.json'))


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Convert the CUHK-03 dataset into the uniform format")
    parser.add_argument(
        'cuhk03_dir',
        help="Root directory of the CUHK-03 dataset containing cuhk-03.mat")
    parser.add_argument(
        'output_dir',
        help="Output directory for the formatted CUHK-03 dataset")
    args = parser.parse_args()
    main(args)