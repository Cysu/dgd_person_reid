import shutil
import numpy as np
from argparse import ArgumentParser
from scipy.misc import imsave

from utils import *


def main(args):
    output_dir = args.output_dir
    mkdir_if_missing(osp.join(args.output_dir, 'cam_0'))
    mkdir_if_missing(osp.join(args.output_dir, 'cam_1'))
    num_identities = 971
    identities = [0] * num_identities
    for i in xrange(num_identities):
        p_images = [[], []]
        for j in xrange(4):
            cam_id = j // 2
            src_file = '{:04d}{:03d}.png'.format(i + 1, j + 1)
            tgt_file = 'cam_{}/{:05d}_{:05d}.png'.format(cam_id, i, j % 2)
            shutil.copy(osp.join(args.cuhk01_dir, src_file),
                        osp.join(args.output_dir, tgt_file))
            p_images[cam_id].append(tgt_file)
        identities[i] = p_images
    # Save meta information into a json file
    meta = {'name': 'cuhk01', 'shot': 'multiple', 'num_cameras': 2}
    meta['identities'] = identities
    write_json(meta, osp.join(output_dir, 'meta.json'))
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
            description="Convert the CUHK-01 dataset into the uniform format")
    parser.add_argument('cuhk01_dir',
            help="Root directory of the CUHK-01 dataset containing image files")
    parser.add_argument('output_dir',
            help="Output directory for the formatted CUHK-01 dataset")
    args = parser.parse_args()
    main(args)