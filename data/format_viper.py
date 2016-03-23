import shutil
import numpy as np
from argparse import ArgumentParser
from glob import glob
from scipy.misc import imsave

from utils import *


def main(args):
    output_dir = args.output_dir
    mkdir_if_missing(osp.join(args.output_dir, 'cam_0'))
    mkdir_if_missing(osp.join(args.output_dir, 'cam_1'))
    identities = []
    cam1_images = glob(osp.join(args.viper_dir, 'cam_a', '*.bmp'))
    cam2_images = glob(osp.join(args.viper_dir, 'cam_b', '*.bmp'))
    cam1_images.sort()
    cam2_images.sort()
    assert len(cam1_images) == len(cam2_images)
    for i in xrange(len(cam1_images)):
        p_id = len(identities)
        p_images = []
        # view-0
        file_name = 'cam_0/{:05d}_{:05d}.bmp'.format(p_id, 0)
        shutil.copy(cam1_images[i],
            osp.join(args.output_dir, file_name))
        p_images.append([file_name])
        # view-1
        file_name = 'cam_1/{:05d}_{:05d}.bmp'.format(p_id, 0)
        shutil.copy(cam2_images[i],
            osp.join(args.output_dir, file_name))
        p_images.append([file_name])
        identities.append(p_images)
    # Save meta information into a json file
    meta = {'name': 'VIPeR', 'shot': 'single', 'num_cameras': 2}
    meta['identities'] = identities
    write_json(meta, osp.join(args.output_dir, 'meta.json'))
    # Randomly create a training and test split
    num = len(identities)
    pids = np.random.permutation(num)
    trainval_pids = sorted(pids[:num // 2])
    test_pids = sorted(pids[num // 2:])
    split = {'trainval': trainval_pids,
             'test_probe': test_pids,
             'test_gallery': test_pids}
    write_json(split, osp.join(output_dir, 'split.json'))


if __name__ == '__main__':
    parser = ArgumentParser(
            description="Convert the VIPeR dataset into the uniform format")
    parser.add_argument('viper_dir',
            help="Root directory of the VIPeR dataset containing "
                 "cam_a/ and cam_b/")
    parser.add_argument('output_dir',
            help="Output directory for the formatted VIPeR dataset")
    args = parser.parse_args()
    main(args)