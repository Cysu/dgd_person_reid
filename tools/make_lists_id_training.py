import numpy as np
from argparse import ArgumentParser

from utils import *


def _get_list(identities):
    ret = []
    for views in identities:
        for v in views:
            for file in v:
                label = int(osp.basename(file)[:5])
                ret.append((file, label))
    return np.asarray(ret)


def _save(file_label_list, file_path):
    content = ['{} {}'.format(x, y) for x, y in file_label_list]
    write_list(content, file_path)


def main(args):
    meta = read_json(osp.join(args.dataset_dir, 'meta.json'))
    split = read_json(osp.join(args.dataset_dir, 'split.json'))
    identities = np.asarray(meta['identities'])
    # Make train / val. Consider single or multiple shot.
    trainval = identities[split['trainval']]
    if meta['shot'] == 'single':
        # When single shot, to ensure each identity has at least one training
        # image, we first randomly choose validation identities, then randomly
        # split their views equally for training and validation.
        num_val = int(len(trainval) * args.val_ratio) * 2
        np.random.shuffle(trainval)
        train = list(trainval[num_val:])
        val = []
        for views in trainval[:num_val]:
            np.random.shuffle(views)
            train.append(views[:len(views) // 2])
            val.append(views[len(views) // 2:])
        train = _get_list(train)
        val = _get_list(val)
    else:
        # When multiple shots, we just randomly split the trainval images
        trainval = _get_list(trainval)
        np.random.shuffle(trainval)
        num_val = int(len(trainval) * args.val_ratio)
        train = trainval[num_val:]
        val = trainval[:num_val]
    # Make test probe / gallery. Probe identities should be a subset of
    # gallery's. First half views are probe, others are gallery.
    assert len(set(split['test_probe']) - set(split['test_gallery'])) == 0
    test_probe, test_gallery = [], []
    for views in identities[split['test_probe']]:
        test_probe.append(views[:len(views) // 2])
        test_gallery.append(views[len(views) // 2:])
    only_in_gallery = list(
            set(split['test_gallery']) - set(split['test_probe']))
    test_gallery.extend(identities[only_in_gallery])
    test_probe = _get_list(test_probe)
    test_gallery = _get_list(test_gallery)
    # Save to files
    mkdir_if_missing(args.output_dir)
    _save(train, osp.join(args.output_dir, 'train.txt'))
    _save(val, osp.join(args.output_dir, 'val.txt'))
    _save(test_probe, osp.join(args.output_dir, 'test_probe.txt'))
    _save(test_gallery, osp.join(args.output_dir, 'test_gallery.txt'))


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Create lists of image file and label for making lmdbs")
    parser.add_argument(
        'dataset_dir',
        help="Directory of a formatted dataset")
    parser.add_argument(
        'output_dir',
        help="Output directory for the lists")
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help="Ratio between validation and trainval data. Default 0.2.")
    args = parser.parse_args()
    main(args)