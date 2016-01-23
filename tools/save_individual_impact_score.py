import sys
import os.path as osp
import lmdb
import shutil
import numpy as np
from argparse import ArgumentParser

from utils import *

if 'external/caffe/python' not in sys.path:
    sys.path.insert(0, 'external/caffe/python')
import caffe
from caffe.proto.caffe_pb2 import Datum


def main(args):
    impact = np.load(args.input_npy)
    assert impact.ndim == 1, "The impact score should be a vector."
    # Create a datum and copy the impact values along the channel
    datum = Datum()
    datum.channels = len(impact)
    datum.height = 1
    datum.width = 1
    del datum.float_data[:]
    datum.float_data.extend(list(impact))
    # Put into lmdb
    if osp.isdir(args.output_lmdb): shutil.rmtree(args.output_lmdb)
    with lmdb.open(args.output_lmdb, map_size=1099511627776) as db:
        with db.begin(write=True) as txn:
            txn.put('impact', datum.SerializeToString())


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Save neurons impact score for an individual domain")
    parser.add_argument('input_npy')
    parser.add_argument('output_lmdb')
    args = parser.parse_args()
    main(args)