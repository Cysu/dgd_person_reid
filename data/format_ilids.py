import shutil
import numpy as np
from argparse import ArgumentParser
from glob import glob
from scipy.misc import imsave
from collections import defaultdict

from utils import *


def main(args):
    output_dir = args.output_dir
    mkdir_if_missing(osp.join(args.output_dir, 'cam_0'))
    mkdir_if_missing(osp.join(args.output_dir, 'cam_1'))
    # Collect the images of each person into dict
    images = glob(osp.join(args.ilids_dir, 'Persons', '*.jpg'))
    pdict = defaultdict(list)
    for imname in images:
        pid = int(osp.basename(imname)[:4])
        pdict[pid].append(imname)
    # Randomly choose half of the images as cam_0, others as cam_1
    identities = []
    for i, (pid, images) in enumerate(pdict.iteritems()):
        num = len(images)
        np.random.shuffle(images)
        p_images = [[], []]
        for src_file in images[:(num // 2)]:
            tgt_file = 'cam_0/{:05d}_{:05d}.jpg'.format(i, len(p_images[0]))
            shutil.copy(src_file, osp.join(args.output_dir, tgt_file))
            p_images[0].append(tgt_file)
        for src_file in images[(num // 2):]:
            tgt_file = 'cam_1/{:05d}_{:05d}.jpg'.format(i, len(p_images[1]))
            shutil.copy(src_file, osp.join(args.output_dir, tgt_file))
            p_images[1].append(tgt_file)
        identities.append(p_images)
    # Save meta information into a json file
    meta = {'name': 'i-LIDS', 'shot': 'multiple', 'num_cameras': 2}
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
            description="Convert the i-LIDS dataset into the uniform format")
    parser.add_argument('ilids_dir',
            help="Root directory of the i-LIDS dataset containing Persons/")
    parser.add_argument('output_dir',
            help="Output directory for the formatted i-LIDS dataset")
    args = parser.parse_args()
    main(args)