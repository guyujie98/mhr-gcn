import torch
from deeprobust.graph.utils import preprocess
from dataset import Dataset
from attacked_data import PrePtbDataset
import numpy as np
from MHRGCN_model import MHRGCN
from MHRGAT_model import MHRGAT
from load_2hop_network import get_2hop_network
import argparse
import warnings
from torch_sparse.tensor import SparseTensor
from deeprobust.graph.utils import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--conduct_attack', action='store_true',
                    default=False, help='choose conduct attacks or not')
parser.add_argument('--attack', type=str, default='meta', choices=['meta', 'sga', 'random'], help='attack')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed'],
                    help='dataset')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--epochs', type=int, default=201)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--ptb_rate', type=float, default=0.25, help="noise ptb_rate")
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--runs', type=int, default=10,
                    help='run how many times')
parser.add_argument('--threshold', type=float, default=0.1,
                    help='threshold tau in MHR-GCN')
args = parser.parse_args()
print(args)
acc = []
for seed in range(1, args.runs + 1):
    print('run ', seed, ' time')
    if args.device != 'cpu':
        torch.cuda.manual_seed(seed)
    data = Dataset(root='./dataset/', name=args.dataset, setting='prognn')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    if args.conduct_attack == False:
        perturbed_adj = adj
        if args.attack == 'sga':
            perturbed_data = PrePtbDataset(root='./dataset/', name=args.dataset, attack_method=args.attack,
                                           ptb_rate=args.ptb_rate)
            idx_test = perturbed_data.target_nodes
            print("performance of MHR-GCN on target nodes under no attack!")
    else:
        if args.attack == 'meta':
            perturbed_data = PrePtbDataset(root='./dataset/', name=args.dataset, attack_method=args.attack,
                                           ptb_rate=args.ptb_rate)
            perturbed_adj = perturbed_data.adj
        if args.attack == 'sga':
            perturbed_data = PrePtbDataset(root='./dataset/', name=args.dataset, attack_method=args.attack,
                                           ptb_rate=args.ptb_rate)
            perturbed_adj = perturbed_data.adj
            idx_test = perturbed_data.target_nodes
        if args.attack == 'random':
            from deeprobust.graph.global_attack import Random
            import random;
            random.seed(seed)
            np.random.seed(seed)
            attacker = Random()
            n_perturbations = int(args.ptb_rate * (adj.sum() // 2))
            attacker.attack(adj, n_perturbations, type='flip')
            perturbed_adj = attacker.modified_adj

    np.random.seed(seed)
    torch.manual_seed(seed)

    perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False)

    adj = perturbed_adj.to(args.device)
    adj2 = get_2hop_network(adj, args.device)
    adj = SparseTensor.from_dense(adj)
    features = features.to(args.device)
    labels = labels.to(args.device)
    classifier = MHRGCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout,
                        threshold=args.threshold,
                        lr=args.lr, device=args.device)
    #test MHRGAT
    # classifier = MHRGAT(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout,
    #                     threshold=args.threshold,
    #                     lr=args.lr, device=args.device)
    classifier = classifier.to(args.device)
    classifier.fit(features, adj, adj2, labels, idx_train, train_iters=args.epochs + 1,
                   idx_val=idx_val,
                   idx_test=idx_test,
                   verbose=True)
    classifier.eval()
    acc_ = classifier.test(idx_test)
    acc.append(acc_.cpu())
print('the average result:', np.mean(acc))
if args.runs != 1:
    print('the standard deviation:', np.std(acc, ddof=1))
