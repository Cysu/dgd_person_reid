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
    split = read_json(osp.join(args.dataset_dir,
            'split_{:02d}.json'.format(args.split_index)))
    identities = np.asarray(meta['identities'])
    # Make train / val. Consider single or multiple shot.
    trainval = identities[split['trainval']]
    if meta['shot'] == 'single':
        # When single shot, to ensure each identity has at least one training
        # image, we first randomly choose validation identities, then randomly
        # split their views equally for training and validation.
        num_val = int(len(trainval) * args.val_ratio) * 2
        np.random.shuffle(trainval)
        train = trainval[num_val:]
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
    # Make test query / probe. Query identities should be a subset of probe's.
    # First half views are query, others are probe.
    assert len(set(split['test_query']) - set(split['test_probe'])) == 0
    test_query, test_probe = [], []
    for views in identities[split['test_query']]:
        test_query.append(views[:len(views) // 2])
        test_probe.append(views[len(views) // 2:])
    only_in_probe = list(set(split['test_probe']) - set(split['test_query']))
    test_probe.extend(identities[only_in_probe])
    test_query = _get_list(test_query)
    test_probe = _get_list(test_probe)
    # Save to files
    mkdir_if_missing(args.output_dir)
    _save(train, osp.join(args.output_dir, 'train.txt'))
    _save(val, osp.join(args.output_dir, 'val.txt'))
    _save(test_query, osp.join(args.output_dir, 'test_query.txt'))
    _save(test_probe, osp.join(args.output_dir, 'test_probe.txt'))


if __name__ == '__main__':
    parser = ArgumentParser(
            description="Create lists of image file and label for making lmdbs")
    parser.add_argument('dataset_dir', help="Directory of a formatted dataset")
    parser.add_argument('output_dir', help="Output directory for the lists")
    parser.add_argument('--split-index', type=int, default=0,
            help="The index of the protocol split to be used")
    parser.add_argument('--val-ratio', type=float, default=0.2,
            help="Ratio between validation and trainval data. Default 0.2.")
    args = parser.parse_args()
    main(args)