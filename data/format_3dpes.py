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
    # Collect the person_id and view_id into dict
    images = glob(osp.join(args.input_dir, 'RGB', '*.bmp'))
    pdict = defaultdict(lambda: defaultdict(list))
    for imname in images:
        pid, vid = osp.basename(imname).split('_')[0:2]
        pdict[pid][vid].append(imname)
    # Randomly choose half of the views as cam_0, others as cam_1
    identities = []
    for i, pid in enumerate(pdict):
        vids = pdict[pid].keys()
        num_views = len(vids)
        np.random.shuffle(vids)
        p_images = [[], []]
        for vid in vids[:(num_views // 2)]:
            for src_file in pdict[pid][vid]:
                tgt_file = 'cam_0/{:05d}_{:05d}.bmp'.format(i, len(p_images[0]))
                shutil.copy(src_file, osp.join(args.output_dir, tgt_file))
                p_images[0].append(tgt_file)
        for vid in vids[(num_views // 2):]:
            for src_file in pdict[pid][vid]:
                tgt_file = 'cam_1/{:05d}_{:05d}.bmp'.format(i, len(p_images[1]))
                shutil.copy(src_file, osp.join(args.output_dir, tgt_file))
                p_images[1].append(tgt_file)
        identities.append(p_images)
    # Save meta information into a json file
    meta = {'name': '3DPeS', 'shot': 'multiple', 'num_cameras': 2}
    meta['identities'] = identities
    write_json(meta, osp.join(args.output_dir, 'meta.json'))
    # Randomly create a training and test split
    num = len(identities)
    pids = np.random.permutation(num)
    trainval_pids = sorted(pids[100:])
    test_pids = sorted(pids[:100])
    split = {'trainval': trainval_pids,
             'test_probe': test_pids,
             'test_gallery': test_pids}
    write_json(split, osp.join(output_dir, 'split.json'))


if __name__ == '__main__':
    parser = ArgumentParser(
            description="Convert the 3DPeS dataset into the uniform format")
    parser.add_argument('input_dir',
            help="Root directory of the 3DPeS dataset containing "
                 "RGB/")
    parser.add_argument('output_dir',
            help="Output directory for the formatted 3DPeS dataset")
    args = parser.parse_args()
    main(args)