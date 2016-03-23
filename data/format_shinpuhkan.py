import shutil
import numpy as np
from argparse import ArgumentParser
from glob import glob
from scipy.misc import imsave

from utils import *


def main(args):
    # cam_0 to cam_15
    for i in xrange(16):
        mkdir_if_missing(osp.join(args.output_dir, 'cam_' + str(i)))
    images = glob(osp.join(args.shinpuhkan_dir, 'images', '*.jpg'))
    images.sort()
    identities = []
    prev_pid = -1
    for name in images:
        name = osp.basename(name)
        p_id = int(name[0:3]) - 1
        c_id = int(name[4:6]) - 1
        if prev_pid != p_id:
            identities.append([])
            prev_cid = -1
        p_images = identities[-1]
        if prev_cid != c_id:
            p_images.append([])
        v_images = p_images[-1]
        file_name = 'cam_{}/{:05d}_{:05d}.jpg'.format(c_id, p_id, len(v_images))
        shutil.copy(osp.join(args.shinpuhkan_dir, 'images', name),
                    osp.join(args.output_dir, file_name))
        v_images.append(file_name)
        prev_pid = p_id
        prev_cid = c_id
    # Save meta information into a json file
    meta = {'name': 'Shinpuhkan', 'shot': 'multiple', 'num_cameras': 16}
    meta['identities'] = identities
    write_json(meta, osp.join(args.output_dir, 'meta.json'))
    # We don't test on this dataset. Just use all the data for train / val.
    split = {'trainval': range(len(identities)),
             'test_probe': [],
             'test_gallery': []}
    write_json(split, osp.join(args.output_dir, 'split.json'))


if __name__ == '__main__':
    parser = ArgumentParser(
            description="Convert the Shinpuhkan dataset into the uniform format")
    parser.add_argument('shinpuhkan_dir',
            help="Root directory of the Shinpuhkan dataset containing images/")
    parser.add_argument('output_dir',
            help="Output directory for the formatted Shinpuhkan dataset")
    args = parser.parse_args()
    main(args)