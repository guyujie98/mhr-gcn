import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import numpy as np
from torch_sparse.tensor import SparseTensor
from deeprobust.graph.utils import *
from torch_geometric.nn import GCNConv
from deeprobust.graph.defense import RGCN
from torch_scatter.scatter import scatter_add, scatter_min
from torch.nn import Sequential, Linear, ReLU
from sklearn.preprocessing import normalize
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix

'''server 9'''


class MHRGCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True,
                 with_bias=True, device=None, threshold=0.1):
        super(MHRGCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.dropout = dropout
        self.lr = lr
        self.threshold = threshold
        self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.adj_norm2 = None
        self.features = None
        self.attention = Attention(nclass, nclass)
        nclass = int(nclass)
        """GCN from geometric"""
        """network from torch-geometric, """
        self.gc1 = GCNConv(nfeat, nhid, bias=True, add_self_loops=True, normalize=True)
        self.gc2 = GCNConv(nhid, nclass, bias=True, add_self_loops=True, normalize=True)

    def forward(self, x, adj, adj2):
        # the first layer
        adj2 = self.att_coef(x, adj2)
        adj = self.att_coef(x, adj)
        x2 = self.gc1(x, adj2)
        x1 = self.gc1(x, adj)

        x1 = F.relu(x1)
        x2 = F.relu(x2)

        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.dropout(x2, self.dropout, training=self.training)

        adj2 = self.att_coef(x2, adj2)
        adj = self.att_coef(x1, adj)

        x2 = self.gc2(x2, adj2)
        x1 = self.gc2(x1, adj)

        x = torch.stack([x1, x2], dim=1)
        x = self.attention(x)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def att_coef(self, features, adj):
        with torch.no_grad():
            row, col = adj.coo()[:2]
            n_total = features.size(0)
            sims = F.cosine_similarity(features[row], features[col])
            mask = sims >= self.threshold
            row = row[mask]
            col = col[mask]
            sims = sims[mask]
            graph_size = torch.Size((n_total, n_total))
            new_adj = SparseTensor(row=row, col=col, value=sims, sparse_sizes=graph_size)
        return new_adj

    def fit(self, features, adj, adj2, labels, idx_train, idx_val=None, idx_test=None, train_iters=81,
            initialize=True,
            verbose=False, patience=1500, ):
        '''
            train the gcn model, when idx_val is not None, pick the best model
            according to the validation loss
        '''
        if initialize:
            self.initialize()

        """The normalization gonna be done in the GCNConv"""
        self.adj_norm = adj
        self.features = features
        self.labels = labels
        self.adj_norm2 = adj2
        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm, self.adj_norm2)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train],
                                    weight=None)  # this weight is the weight of each training nodes
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm, self.adj_norm2)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm, self.adj_norm2)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            self.eval()
            output = self.forward(self.features, self.adj_norm, self.adj_norm2)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if verbose and i % 20 == 0:
                print('Epoch {},training loss: {}, val loss: {},val acc: {}, '.format(i, loss_train.item(), loss_val,
                                                                                      acc_val))

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
        print('best val:', best_loss_val, best_acc_val)
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm, self.adj_norm2)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.adj_norm, self.adj_norm2)

            if verbose and i % 20 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
            print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val))
        self.load_state_dict(weights)

    def test(self, idx_test):
        self.eval()
        output = self.predict(self.features, self.adj_norm,
                              self.adj_norm2)  # here use the self.features and self.adj_norm in training stage
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test

    def _set_parameters(self):
        # TODO
        pass

    def predict(self, features=None, adj=None, adj2=None, ):
        '''By default, inputs are unnormalized data'''
        self.eval()
        return self.forward(features, adj, adj2)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        sparserow = torch.LongTensor(sparse_mx.row).unsqueeze(1)
        sparsecol = torch.LongTensor(sparse_mx.col).unsqueeze(1)
        sparseconcat = torch.cat((sparserow, sparsecol), 1)
        sparsedata = torch.FloatTensor(sparse_mx.data)
        return torch.sparse.FloatTensor(sparseconcat.t(), sparsedata, torch.Size(sparse_mx.shape))


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(in_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.W1.reset_parameters()
        self.W2.reset_parameters()

    def forward(self, z):
        w = self.W1(z)
        w = F.tanh(w)
        w = self.W2(w)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)
