import numpy as np
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances

from utils import *


def _get_train_data(result_dir):
    # Merge training and validation features and labels
    features = np.r_[np.load(osp.join(result_dir, 'train_features.npy')),
                     np.load(osp.join(result_dir, 'val_features.npy'))]
    labels = np.r_[np.load(osp.join(result_dir, 'train_labels.npy')),
                   np.load(osp.join(result_dir, 'val_labels.npy'))]
    # Reassign the labels to make them sequentially numbered from zero
    unique_labels = np.unique(labels)
    labels_map = {l: i for i, l in enumerate(unique_labels)}
    labels = np.asarray([labels_map[l] for l in labels])
    return features, labels


def _get_test_data(result_dir):
    PX = np.load(osp.join(result_dir, 'test_probe_features.npy'))
    PY = np.load(osp.join(result_dir, 'test_probe_labels.npy'))
    GX = np.load(osp.join(result_dir, 'test_gallery_features.npy'))
    GY = np.load(osp.join(result_dir, 'test_gallery_labels.npy'))
    # Reassign the labels to make them sequentially numbered from zero
    unique_labels = np.unique(np.r_[PY, GY])
    labels_map = {l: i for i, l in enumerate(unique_labels)}
    PY = np.asarray([labels_map[l] for l in PY])
    GY = np.asarray([labels_map[l] for l in GY])
    return PX, PY, GX, GY


def _learn_pca(X, dim):
    pca = PCA(n_components=dim)
    pca.fit(X)
    return pca


def _learn_metric(X, Y, method):
    if method == 'euclidean':
        M = np.eye(X.shape[1])
    elif method == 'kissme':
        num = len(Y)
        X1, X2 = np.meshgrid(np.arange(0, num), np.arange(0, num))
        X1, X2 = X1[X1 < X2], X2[X1 < X2]
        matches = (Y[X1] == Y[X2])
        num_matches = matches.sum()
        num_non_matches = len(matches) - num_matches
        idxa = X1[matches]
        idxb = X2[matches]
        S = X[idxa] - X[idxb]
        C1 = S.transpose().dot(S) / num_matches
        p = np.random.choice(num_non_matches, num_matches, replace=False)
        idxa = X1[matches == False]
        idxb = X2[matches == False]
        idxa = idxa[p]
        idxb = idxb[p]
        S = X[idxa] - X[idxb]
        C0 = S.transpose().dot(S) / num_matches
        M = np.linalg.inv(C1) - np.linalg.inv(C0)
    return M


def _eval_cmc(PX, PY, GX, GY, M):
    D = pairwise_distances(GX, PX, metric='mahalanobis', VI=M, n_jobs=-2)
    C = cmc(D, GY, PY)
    return C


def main(args):
    X, Y = _get_train_data(args.result_dir)
    PX, PY, GX, GY = _get_test_data(args.result_dir)

    file_suffix = args.method
    if args.pca is not None:
        S = _learn_pca(X, args.pca)
        X = S.transform(X)
        PX = S.transform(PX)
        GX = S.transform(GX)
        file_suffix += '_pca_{}'.format(args.pca)
        np.save(osp.join(args.result_dir, 'pca_' + file_suffix), S)

    M = _learn_metric(X, Y, args.method)
    C = _eval_cmc(PX, PY, GX, GY, M)
    np.save(osp.join(args.result_dir, 'metric_' + file_suffix), M)
    np.save(osp.join(args.result_dir, 'cmc_' + file_suffix), C)

    for topk in [1, 5, 10, 20]:
        print "{:8}{:8.1%}".format('top-' + str(topk), C[topk - 1])

if __name__ == '__main__':
    parser = ArgumentParser(
            description="Metric learning and evaluate performance")
    parser.add_argument('result_dir',
            help="Result directory. Containing extracted features and labels. "
                 "CMC curve will also be saved to this directory.")
    parser.add_argument('--method', choices=['euclidean', 'kissme'],
            default='euclidean')
    parser.add_argument('--pca', type=int,
            help="If specified, will reduce features to this dimension by PCA")
    args = parser.parse_args()
    main(args)