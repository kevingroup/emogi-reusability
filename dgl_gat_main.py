import sys
import os

import argparse
import random
import numpy as np
import torch
import dgl
from GAT import GATModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random generator seed.', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--hidden_dims', type=int, default=128, help='hidden dims for head')
    parser.add_argument('--loss_mul', type=int, default=1)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')
    parser.add_argument("--print_every", type=int, default=10, help='eval_interval')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--heads', type=int, default=4, help='Num of heads for GAT layer')
    parser.add_argument('--sample_filename', type=str, default='/research/dept4/cyhong/cancer_gcn/EMOGI/saved_model/EMOGI_CPDB/CPDB_multiomics.h5')
    parser.add_argument('--cuda', default=False, action="store_true")

    args = parser.parse_args()
    return args


def set_seed(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.multiprocessing.set_sharing_strategy('file_system')
    dgl.random.seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    return device


if __name__ == '__main__':
    args = get_args()
    device = set_seed(args)

    model = GATModel(args, device=device)
    model.learning()
    
