import sys
import os
import os.path as osp
import numpy as np
import google.protobuf as pb
from argparse import ArgumentParser

from utils import *


if 'external/caffe/python' not in sys.path:
    sys.path.insert(0, 'external/caffe/python')
import caffe
from caffe.proto.caffe_pb2 import NetParameter


def parse_prototxt(model_file, layer_name):
    with open(model_file) as fp:
        net = NetParameter()
        pb.text_format.Parse(fp.read(), net)
    for i, layer in enumerate(net.layer):
        if layer.name != layer_name: continue
        blob = layer.top[0]
        for j in xrange(i + 1, len(net.layer)):
            if blob in net.layer[j].bottom:
                next_layer = net.layer[j].name
                return blob, next_layer
    raise ValueError(
        "Cannot find layer {} or its next layer".format(layer_name))


def main(args):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    blob, next_layer = parse_prototxt(args.model, args.layer)
    net = caffe.Net(args.model, args.weights, caffe.TEST)
    # Channelwise for conv
    impact = np.zeros(net.blobs[blob].shape[1])
    for i in xrange(args.num_iters):
        net.forward()
        f = net.blobs[blob].data.copy()
        loss = net.blobs['loss'].data.copy()
        for n in xrange(f.shape[1]):
            net.blobs[blob].data[...] = f.copy()
            net.blobs[blob].data[:, n] = 0
            net.forward(start=next_layer)
            delta = net.blobs['loss'].data - loss
            impact[n] += delta.sum()
    # Normalize
    if args.normalize:
        assert impact.max() > 0, "No neuron has positive impact"
        scale = np.log(9) / impact.max()
        impact *= scale
    else:
        batch_size = net.blobs[blob].shape[0]
        impact /= (batch_size * args.num_iters)
    # Save
    np.save(args.output, impact)


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Compute neurons impact on a particular domain")
    parser.add_argument('model')
    parser.add_argument('weights')
    parser.add_argument('output')
    parser.add_argument('--num_iters', type=int, required=True)
    parser.add_argument('--layer', type=str, default='fc7')
    parser.add_argument('--normalize', action='store_true',
        help="Normalize to make sigmoid(highest impact) == 0.9")
    args = parser.parse_args()
    main(args)