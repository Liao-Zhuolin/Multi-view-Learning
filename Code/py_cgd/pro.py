import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import scipy.io as scio
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

def NormalizeFea(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array((features**2).sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def knn_graph(X, k, threshold):
    """
    KNN_GRAPH Construct W using KNN graph
    :param X: data point features, n-by-p maxtirx.
    :param k: number of nn.
    :param threshold: distance threshold.
    :return: adjacency matrix, n-by-n matrix.
    """
    dist = np.linalg.norm(np.expand_dims(X, 0) - np.expand_dims(X, 1), axis=-1)
    dist_argsort = np.argsort(dist, axis=-1)
    mask = np.zeros_like(dist).astype(bool)
    for idx, c in enumerate(dist_argsort):
        mask[idx, c[1:k + 1]] = True
    mask &= dist < threshold
    dist = np.exp(-dist ** 2)
    dist[~mask] = 0
    dist = np.maximum(dist.T, dist)
    return dist

def load_data(dataFile): # {multi-view}
    """Load data."""
    data = scio.loadmat(dataFile)
    features = data['X'][0]
    labels = data['Y']
    #adj = []
    #for i in range(0,len(features)):
        #print(max([item for sublist in features[i] for item in sublist]))
        #features[i] = preprocess_features(features[i])
    #for i in range(0,len(features)):
        #adj.append(np.dot(preprocess_features(features[i]), preprocess_features(features[i]).T))
    #    adj.append(knn_graph(features[i], 6, 0.1))
    return features, labels

def bestMap(L1,L2):
    '''
    bestmap: permute labels of L2 to match L1 as good as possible
        INPUT:  
            L1: labels of L1, shape of (N,) vector
            L2: labels of L2, shape of (N,) vector
        OUTPUT:
            new_L2: best matched permuted L2, shape of (N,) vector
    version 1.0 --December/2018
    Modified from bestMap.m (written by Deng Cai)
    '''

    if L1.shape[0] != L2.shape[0] or len(L1.shape) > 1 or len(L2.shape) > 1: 
        raise Exception('L1 shape must equal L2 shape')
        return 

    Label1 = np.unique(L1)
    nClass1 = Label1.shape[0]
    Label2 = np.unique(L2)
    nClass2 = Label2.shape[0]
    nClass = max(nClass1,nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[j,i] = np.sum((np.logical_and(L1 == Label1[i], L2 == Label2[j])).astype(np.int64))
    c,t = linear_sum_assignment(-G)
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[t[i]]
    return newL2

def spectral(W, k):
    """
    SPECTRUAL spectral clustering
    :param W: Adjacency matrix, N-by-N matrix
    :param k: number of clusters
    :return: data point cluster labels, n-by-1 vector.
    """
    w_sum = np.array(W.sum(axis=1)).reshape(-1)
    D = np.diag(w_sum)
    _D = np.diag(w_sum ** (-1 / 2))
    L = D - W
    L = _D @ L @ _D
    eigval, eigvec = np.linalg.eig(L)
    eigval_argsort = eigval.real.astype(np.float32).argsort()
    F = np.take(eigvec.real.astype(np.float32), eigval_argsort[:k], axis=-1)
    idx = KMeans(n_clusters=k).fit(F).labels_
    return idx

def bs_convert2sim_knn(dist, K, sigma):
    dist = dist/np.max(np.max(dist, 1))
    sim = np.exp(-dist**2/(sigma**2))
    if K>0:
        idx = sim.argsort()[:,::-1]
        sim_new = np.zeros_like(sim)
        for ii in range(0, len(sim_new)):
            sim_new[ii, idx[ii,0:K]] = sim[ii, idx[ii,0:K]]
        sim = (sim_new + sim_new.T)/2
    else:
        sim = (sim + sim.T)/2
    return sim

def purity(cluster, label):
    cluster = np.array(cluster)
    label = np. array(label)
    indedata1 = {}
    for p in np.unique(label):
        indedata1[p] = np.argwhere(label == p)
    indedata2 = {}
    for q in np.unique(cluster):
        indedata2[q] = np.argwhere(cluster == q)

    count_all = []
    for i in indedata1.values():
        count = []
        for j in indedata2.values():
            a = np.intersect1d(i, j).shape[0]
            count.append(a)
        count_all.append(count)

    return sum(np.max(count_all, axis=0))/len(cluster)

def evaluate(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)
    Precision = metrics.precision_score(label, pred, average='macro')
    Recall = metrics.recall_score(label, pred, average='macro')
    Purity = purity(label, pred)
    return nmi, ari, f, Precision, Recall, Purity