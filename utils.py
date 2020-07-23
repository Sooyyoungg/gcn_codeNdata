import torch
import numpy as np;
from torch.autograd import Variable
from graph import *;
import math;
from numpy import linalg as LA
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
#from dataset import data_handling

import sklearn.datasets
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import scipy.sparse as sp
from sklearn.preprocessing import normalize


class computeLaplacian:
    def __init__(self,X_train,args,dist):
        # X, fs_labels, cov, y, groups = data_handling.create_data_set(complete=False, completeness_threshold=0.9,
        #                        covariates=['Age', 'Sex', 'age_group_tesla_site'], min_counts_per_site='auto')
        # self.subset = subset

        # if self.subset == 'train':
        self.data = np.asarray(X_train)
        self.data = normalize(self.data, axis=0, norm='l1')

        # print(self.data)
        if args.model == 'GCN' or args.model == 'ChebyNet':
            self.compute_laplacian(args,dist)
        if args.model == 'DPIEnn':
            self.compute_dpie(args)
        if args.model == 'MoNet' or args.model == 'DSGC':
            self.compute_pseudo(args,dist)
        # if args.model == 'DSGC':
        self.w = 1
        self.m = X_train.shape[1]

    def dense_to_sp(self, adj):
        """transform tensor adjacency matrix to sparse tensor in PyTorch"""
        adj = sp.coo_matrix(adj.numpy())
        values = adj.data
        indices = np.vstack((adj.row, adj.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adj.shape
        adj = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        return adj

    def compute_dpie(self, args):
        print('==> Computing DPIE..')
        self.dpie_list = []
        self.dpiv_list = []
        self.embedding_num = args.embedding

        # for i in range(1):
        DRev, NormF, DRevsqrt = prep_cos(self.data.T)
        u, uo, v = dpie(self.data.T, NormF, DRev, DRevsqrt, 1e-8, 100, self.embedding_num+30, self.embedding_num, 1e-8, 'rw')
        self.dpie_list += [torch.from_numpy(uo.astype(np.float32))]
        self.dpiv_list += [torch.from_numpy(v.astype(np.float32))]

    def compute_laplacian(self, args,dist):
        print('==> Computing Laplacian..')
        #dist, nn_list = distance_sklearn_metrics(self.data.T, k=args.nn)
        dist, nn_list = distance_sklearn_metrics(dist, k=args.nn)
        L_gcn = adjacency_gcn(dist, nn_list,sigma=args.sigma)
        self.L_gcn = L_gcn #self.dense_to_sp(L_gcn)
        L_cheby = adjacency_cheby(dist, nn_list)
        self.L_cheby = L_cheby#self.dense_to_sp(L_cheby)


    def compute_pseudo(self, args,dist):
        print('==> Computing Pseudo Coordinates..')
        self.dist, self.nn_list = distance_sklearn_metrics(dist, k=args.nn)
        # self.dist, self.nn_list = distance_sklearn_metrics(self.data.T, k=args.nn)
        self.L_idx, self.pseudo = adjacency_2d(self.dist, self.nn_list, args.nn)
