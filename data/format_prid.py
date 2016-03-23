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
    # Randomly choose 100 people from the 200 shared people as test probe
    p = list(np.random.permutation(200))
    test_probe = range(100)
    test_gallery = range(100)
    identities = []
    for pid in p[:100]:
        p_images = []
        src_file = osp.join(args.prid_dir, 'single_shot', 'cam_a',
                            'person_{:04d}.png'.format(pid + 1))
        tgt_file = osp.join('cam_0', '{:05d}_00000.png'.format(len(identities)))
        shutil.copy(src_file, osp.join(args.output_dir, tgt_file))
        p_images.append([tgt_file])
        src_file = osp.join(args.prid_dir, 'single_shot', 'cam_b',
                            'person_{:04d}.png'.format(pid + 1))
        tgt_file = osp.join('cam_1', '{:05d}_00000.png'.format(len(identities)))
        shutil.copy(src_file, osp.join(args.output_dir, tgt_file))
        p_images.append([tgt_file])
        identities.append(p_images)
    # Other 100 people from the 200 as a part of trainval
    # Choose 10 images randomly from the multi-shot images
    trainval = range(100, 200)
    for pid in p[100:]:
        p_images = [[], []]
        images = glob(osp.join(args.prid_dir, 'multi_shot', 'cam_a',
                               'person_{:04d}'.format(pid + 1), '*.png'))
        images = np.random.choice(images, size=min(10, len(images)),
                                  replace=False)
        for src_file in images:
            tgt_file = osp.join('cam_0',
                    '{:05d}_{:05d}.png'.format(len(identities), len(p_images[0])))
            shutil.copy(src_file, osp.join(args.output_dir, tgt_file))
            p_images[0].append(tgt_file)
        images = glob(osp.join(args.prid_dir, 'multi_shot', 'cam_b',
                               'person_{:04d}'.format(pid + 1), '*.png'))
        images = np.random.choice(images, size=min(10, len(images)),
                                  replace=False)
        for src_file in images:
            tgt_file = osp.join('cam_1',
                    '{:05d}_{:05d}.png'.format(len(identities), len(p_images[1])))
            shutil.copy(src_file, osp.join(args.output_dir, tgt_file))
            p_images[1].append(tgt_file)
        identities.append(p_images)
    # 201 to 385 cam_a people as another part of trainval
    for pid in xrange(200, 385):
        p_images = [[], []]
        images = glob(osp.join(args.prid_dir, 'multi_shot', 'cam_a',
                               'person_{:04d}'.format(pid + 1), '*.png'))
        images = np.random.choice(images, size=min(10, len(images)),
                                  replace=False)
        for src_file in images:
            tgt_file = osp.join('cam_0',
                    '{:05d}_{:05d}.png'.format(len(identities), len(p_images[0])))
            shutil.copy(src_file, osp.join(args.output_dir, tgt_file))
            p_images[0].append(tgt_file)
        trainval.append(len(identities))
        identities.append(p_images)
    # 201 to 749 cam_b people as additional test gallery
    for pid in xrange(200, 749):
        src_file = osp.join(args.prid_dir, 'single_shot', 'cam_b',
                            'person_{:04d}.png'.format(pid + 1))
        tgt_file = osp.join('cam_1', '{:05d}_00000.png'.format(len(identities)))
        shutil.copy(src_file, osp.join(args.output_dir, tgt_file))
        p_images = [[], [tgt_file]]
        test_gallery.append(len(identities))
        identities.append(p_images)
    # Save meta information into a json file
    meta = {'name': 'PRID', 'shot': 'multiple', 'num_cameras': 2}
    meta['identities'] = identities
    write_json(meta, osp.join(args.output_dir, 'meta.json'))
    # We have only one split
    split = {'trainval': trainval,
             'test_probe': test_probe,
             'test_gallery': test_gallery}
    write_json(split, osp.join(output_dir, 'split.json'))


if __name__ == '__main__':
    parser = ArgumentParser(
            description="Convert the PRID dataset into the uniform format")
    parser.add_argument('prid_dir',
            help="Root directory of the PRID dataset containing "
                 "single_shot/ and multi_shot/")
    parser.add_argument('output_dir',
            help="Output directory for the formatted PRID dataset")
    args = parser.parse_args()
    main(args)