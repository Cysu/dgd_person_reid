import sys
import os.path as osp
import lmdb
import shutil
import numpy as np
from glob import glob
from argparse import ArgumentParser

from utils import *

if 'external/caffe/python' not in sys.path:
    sys.path.insert(0, 'external/caffe/python')
import caffe
from caffe.proto.caffe_pb2 import Datum


def load_domain_impact(impact_dir):
    files = glob(osp.join(impact_dir, '*.npy'))
    domain_datum = {}
    for file_name in files:
        domain_name = osp.splitext(osp.basename(file_name))[0]
        impact = np.load(file_name)
        assert impact.ndim == 1, "The impact score should be a vector."
        datum = Datum()
        datum.channels = len(impact)
        datum.height = 1
        datum.width = 1
        del datum.float_data[:]
        datum.float_data.extend(list(impact))
        domain_datum[domain_name] = datum.SerializeToString()
    return domain_datum


def main(args):
    domain_datum = load_domain_impact(args.impact_dir)
    file_paths = read_list(args.image_list_file)
    if osp.isdir(args.output_lmdb): shutil.rmtree(args.output_lmdb)
    with lmdb.open(args.output_lmdb, map_size=1099511627776) as db:
        with db.begin(write=True) as txn:
            for i, file_path in enumerate(file_paths):
                find_match = False
                for domain, datum in domain_datum.iteritems():
                    if domain not in file_path: continue
                    txn.put('{:010d}_{}'.format(i, domain), datum)
                    find_match = True
                    break
                if not find_match:
                    print "Warning: cannot find the domain of {}".format(
                        file_path)


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Save neurons impact score a joint dataset. Each sample "
                    "has a datum.")
    parser.add_argument('impact_dir',
        help="A directory of numpy files which are named after each domain")
    parser.add_argument('image_list_file',
        help="A txt file of a list of images. KV format is also fine.")
    parser.add_argument('output_lmdb')
    args = parser.parse_args()
    main(args)