import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
import tempfile
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as Func

import ipc_service
import dgl
from dgl.nn.pytorch import SAGEConv
from dgl.heterograph import DGLBlock
import time
import numpy as np

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    if torch.cuda.is_available():
      dist.init_process_group('nccl', rank=rank, world_size=world_size)
    else:
      dist.init_process_group('gloo', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, 'mean'))
        for _ in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

def create_dgl_block(src, dst, num_src_nodes, num_dst_nodes):
    gidx = dgl.heterograph_index.create_unitgraph_from_coo(2, num_src_nodes, num_dst_nodes, src, dst, 'coo')
    g = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])

    return g

def train_one_step(model, optimizer, loss_fcn, device, feat_len):
    
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()
    blocks = []
    blocks.append(create_dgl_block(block1_agg_src, block1_agg_dst, block1_src_num, block1_dst_num))
    blocks.append(create_dgl_block(block2_agg_src, block2_agg_dst, block2_src_num, block2_dst_num))
    batch_pred = model(blocks, features)
    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    loss = loss_fcn(batch_pred, long_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ipc_service.synchronize()

def valid_one_step(model, loss_fcn, device, feat_len):
    
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()
    blocks = []
    blocks.append(create_dgl_block(block1_agg_src, block1_agg_dst, block1_src_num, block1_dst_num))
    blocks.append(create_dgl_block(block2_agg_src, block2_agg_dst, block2_src_num, block2_dst_num))
    batch_pred = model(blocks, features)
    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    loss = loss_fcn(batch_pred, long_labels)
    return loss

def test_one_step(model, metric, device, feat_len):
    
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()
    blocks = []
    blocks.append(create_dgl_block(block1_agg_src, block1_agg_dst, block1_src_num, block1_dst_num))
    blocks.append(create_dgl_block(block2_agg_src, block2_agg_dst, block2_src_num, block2_dst_num))
    batch_pred = model(blocks, features)
    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    acc = metric(batch_pred, long_labels)
    return acc

def worker_process(rank, world_size, args):
    print(f"Running basic DDP example on rank {rank}.")
    device_id = rank
    setup(rank, world_size)
    cuda_device = torch.device("cuda:{}".format(device_id))
    torch.cuda.set_device(cuda_device)
    ipc_service.initialize()
    train_steps, valid_steps, test_steps = ipc_service.get_steps()
    batch_size = (args.train_batch_size)
    hop1 = (args.nbrs_num)[0]
    hop2 = (args.nbrs_num)[1]

    num_ids = batch_size * (1 + hop1 + hop1 * hop2)
    feat_len = args.features_num
    num_feat = num_ids * feat_len

    model = SAGE(in_feats=args.features_num,
                        n_hidden=args.hidden_dim,
                        n_classes=args.class_num,
                        n_layers=args.hops_num,
                        activation=Func.relu,
                        dropout=args.drop_rate).to(cuda_device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[device_id])
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()

    epoch_num = args.epoch

    for epoch in range(epoch_num):
        forward = 0
        start = time.time()
        epoch_time = 0
        for iter in range(train_steps):
            train_one_step(model, optimizer, loss_fcn, cuda_device, feat_len)
        epoch_time += time.time() - start

        loss = []
        for iter in range(valid_steps):
            loss_per_batch = valid_one_step(model, optimizer, loss_fcn, cuda_device, feat_len)
            loss.append(loss_per_batch)
        val_loss = np.mean(loss[:])
        if device_id == 0:
            print('Epoch:{}, Cost:{} s, Loss on CUDA {} :{} '.format(epoch, epoch_time, device_id, val_loss))
    
    print('Finish training, start test')
    model.eval()
    model.metric = metric
    metric = torchmetrics.Accuracy()
    for iter in range(test_steps):
        infer_one_step(model, optimizer, loss_fcn, cuda_device, feat_len)
    acc = metric.compute()
    if device_id == 0:
        print("Accuracy on test data: {acc}")
    metric.reset()

    ipc_service.finalize()
    cleanup()

def run_distribute(dist_fn, world_size, args):
    mp.spawn(dist_fn,
             args=(world_size, args),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    cur_path = sys.path[0]
    argparser = argparse.ArgumentParser("Train Graphsage.")
    argparser.add_argument('--class_num', type=int, default=2)
    argparser.add_argument('--features_num', type=int, default=128)
    argparser.add_argument('--train_batch_size', type=int, default=8000)
    argparser.add_argument('--hidden_dim', type=int, default=256)
    argparser.add_argument('--hops_num', type=int, default=2)
    argparser.add_argument('--nbrs_num', type=list, default=[25, 10])
    argparser.add_argument('--drop_rate', type=float, default=0.0)
    argparser.add_argument('--learning_rate', type=float, default=0.001)
    argparser.add_argument('--epoch', type=int, default=10)
    args = argparser.parse_args()

    # n_gpus = torch.cuda.device_count()
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = 8

    run_distribute(worker_process, world_size, args)
