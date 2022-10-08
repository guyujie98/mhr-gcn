import torch
from torch_sparse import SparseTensor


def get_2hop_network(adj, device='cuda:0'):
    '''remove self-loop'''
    n = adj.size(0)
    adj = adj - torch.eye(n).to(device)
    adj[adj < 0] = 0

    adj2 = adj @ adj
    adj2[adj2 > 1] = 1
    adj2 = adj2 - adj - torch.eye(n).to(device)
    adj2[adj2 < 0] = 0
    adj2 = SparseTensor.from_dense(adj2).to(device)
    print("2-hop network done!")
    return adj2
