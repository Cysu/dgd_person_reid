import numpy as np
from argparse import ArgumentParser

from utils import *


def main(args):
    id_offset = 0
    merged_train_kv = {}
    merged_val_kv = {}
    for dataset_dir, db_dir in zip(args.dataset_dirs, args.db_dirs):
        train_files, train_labels = read_kv(osp.join(db_dir, 'train.txt'))
        val_files, val_labels = read_kv(osp.join(db_dir, 'val.txt'))
        unique_ids = set(map(int, train_labels + val_labels))
        id_mapping = {idx: i + id_offset for i, idx in enumerate(unique_ids)}
        for k, v in zip(train_files, train_labels):
            merged_train_kv[osp.join(dataset_dir, k)] = id_mapping[int(v)]
        for k, v in zip(val_files, val_labels):
            merged_val_kv[osp.join(dataset_dir, k)] = id_mapping[int(v)]
        id_offset += len(id_mapping)
    mkdir_if_missing(osp.join(args.output_dir))
    train_list = [k + ' ' + str(v) for k, v in merged_train_kv.iteritems()]
    np.random.shuffle(train_list)
    write_list(train_list, osp.join(args.output_dir, 'train.txt'))
    write_kv(merged_val_kv.keys(), map(str, merged_val_kv.values()),
             osp.join(args.output_dir, 'val.txt'))
    print "Max ID:", id_offset


if __name__ == '__main__':
    parser = ArgumentParser(
            description="Merge multiple lists of train / val image file and "
                        "label into a single-task one")
    parser.add_argument('--dataset-dirs', type=str, nargs='+',
            help="Dataset directories containing cam_0/, cam_1/, ...")
    parser.add_argument('--db-dirs', type=str, nargs='+',
            help="Database directories containing train.txt and val.txt. "
                 "Must have the same number of dirs with dataset_dirs")
    parser.add_argument('output_dir', help="Output directories for the lists")
    args = parser.parse_args()
    assert len(args.dataset_dirs) == len(args.db_dirs)
    main(args)