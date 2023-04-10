# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
import sys
import time
try:
  import torch
except ImportError:
  pass

import graphlearn as gl
import graphlearn.python.nn.pytorch as thg
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from tqdm import tqdm

from gcn import GCN

os.environ["ODPS_CONFIG_FILE_PATH"] = "odps_config.ini"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_env(args):
  world_size = 1
  rank = 0
  if args.ddp:
    if torch.cuda.is_available():
      dist.init_process_group('nccl')
    else:
      dist.init_process_group('gloo')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
  print('world_size:', world_size, 'rank:', rank)
  args.use_mp = args.client_num > 0
  args.world_size = world_size
  args.rank = rank


def load_graph(args):
  dataset_folder = args.dataset_folder
  node_type = 'item'
  edge_type = 'relation'
  # shoud be split when distributed training.
  node_path = dataset_folder + "node_table"
  edge_path = dataset_folder + "edge_table"
  train_path = dataset_folder + "train_table"
  val_path = dataset_folder + "val_table"
  test_path = dataset_folder + "test_table"


  gl.set_default_label(0)
  g = gl.Graph()                                                           \
        .node(node_path, node_type=node_type,
              decoder=gl.Decoder(labeled=True,
                                 attr_types=["float"] * args.features_num,
                                 attr_delimiter=":"))                      \
        .edge(edge_path,
              edge_type=(node_type, node_type, edge_type),
              decoder=gl.Decoder(weighted=True), directed=False)           \
        .node(train_path, node_type=node_type,
              decoder=gl.Decoder(weighted=True), mask=gl.Mask.TRAIN)       \
        .node(val_path, node_type=node_type,
              decoder=gl.Decoder(weighted=True), mask=gl.Mask.VAL)         \
        .node(test_path, node_type=node_type,
              decoder=gl.Decoder(weighted=True), mask=gl.Mask.TEST)
  return g

def query(g, args, mask=gl.Mask.TRAIN):
  if mask == gl.Mask.TRAIN:
    seed = g.V('item', mask=mask).batch(args.train_batch_size).shuffle(traverse=True).alias('src')
  else:
    seed = g.V('item', mask=mask).batch(args.test_batch_size).alias('src')
  seed.outV('relation').sample(args.nbrs_num).by('full').alias('src_hop1')
  return seed.values()

def induce_func(data_dict):
  """induce the src and 1-hop neighhbor to a list of pyG `Data`."""
  src = data_dict['src']
  nbr = data_dict['src_hop1']
  subgraphs = []
  offset = 0
  for i in range (src.ids.size):
    float_attrs = np.expand_dims(src.float_attrs[i], axis=0)
    labels = np.expand_dims(src.labels[i], axis=0)
    row, col = [], []
    begin, end = offset, offset + nbr.offsets[i]
    float_attrs = np.concatenate((float_attrs, nbr.float_attrs[begin:end]), axis=0)
    labels = np.concatenate((labels, nbr.labels[begin:end]), axis=0)
    for j in range(nbr.offsets[i]):
      row.append(0)
      col.append(j+1)
      col.append(0)
      row.append(j+1)
    offset += nbr.offsets[i]
    edge_index = np.stack([np.array(row), np.array(col)], axis=0)
    subgraph = Data(torch.from_numpy(float_attrs),
                    torch.from_numpy(edge_index).to(torch.long),
                    y=torch.from_numpy(labels).to(torch.long))
    subgraphs.append(subgraph)
  return subgraphs

def train(model, loader, optimizer, args):
  model.train()
  for i, data in tqdm(enumerate(loader)):
    optimizer.zero_grad()
    data = data.to(device)
    x = model(data)
    x = F.log_softmax(x, dim=1)
    loss = F.nll_loss(x, data.y)
    print('loss: ', loss.item())
    loss.backward()
    optimizer.step()

@torch.no_grad()
def test(model, loader, args):
  model.eval()
  y_pred, y_true = [], []
  for i, data in tqdm(enumerate(loader)):
    data = data.to(device)
    y_pred.append(F.log_softmax(model(data), dim=1).cpu().max(1)[1])
    y_true.append(data.y.view(-1).cpu().to(torch.float))
  test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)
  acc = test_pred.eq(test_true).sum().item() / test_true.size()[0]
  return acc

def run(args):
  gl.set_tape_capacity(1)
  g = load_graph(args)
  gl.set_tracker_mode(0)
  if args.use_mp:
    thg.set_client_num(args.client_num)
    thg.launch_server(g)
  else:
    g.init(task_index=args.rank, task_count=args.world_size,
           hosts=thg.get_cluster_spec()['server'])

  # train loader
  train_query = query(g, args, mask=gl.Mask.TRAIN)
  if args.use_mp:
    train_dataset = thg.Dataset(train_query, window=10, induce_func=induce_func, graph=g)
    graph_counts = thg.get_counts()
    while True:
      if not graph_counts:
        time.sleep(1)
      else:
        item_count_per_server = graph_counts[gl.get_mask_type('item', gl.Mask.TRAIN)]
        break
  else:
    train_dataset = thg.Dataset(train_query, window=10, induce_func=induce_func)
    item_count_per_server = g.server_get_stats()[gl.get_mask_type('item', gl.Mask.TRAIN)]

  print('item node count per server: ', item_count_per_server)
  length_per_worker =  min(item_count_per_server) // args.train_batch_size
  print('length_per_worker being set to: ' + str(length_per_worker))
  train_loader = thg.PyGDataLoader(train_dataset, multi_process=args.use_mp, length=length_per_worker)

  # test loader
  test_query = query(g, args, mask=gl.Mask.TEST)
  if args.use_mp:
    test_dataset = thg.Dataset(test_query, window=10, induce_func=induce_func, graph=g)
  else:
    test_dataset = thg.Dataset(test_query, window=10, induce_func=induce_func)
  test_loader = thg.PyGDataLoader(test_dataset, multi_process=args.use_mp)

  # define model
  model = GCN(input_dim=args.features_num,
              hidden_dim=args.hidden_dim,
              output_dim=args.class_num,
              depth=args.depth,
              drop_rate=args.drop_rate).to(device)
  if dist.is_initialized():
    model = torch.nn.parallel.DistributedDataParallel(model)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

  # train and test
  for epoch in range(0, args.epoch):
    train(model, train_loader, optimizer, args)
    test_acc = test(model, test_loader, args)
    log = 'Epoch: {:03d}, Test: {:.4f}'
    print(log.format(epoch, test_acc))

  if not args.use_mp:
    g.close()


if __name__ == "__main__":
  cur_path = sys.path[0]
  argparser = argparse.ArgumentParser("Train GCN.")
  argparser.add_argument('--dataset_folder', type=str,
                         default=os.path.join(cur_path, '../../data/cora/'),
                         help="Dataset Folder, list files are node_table, edge_table, "
                              "train_table, val_table and test_table")
  argparser.add_argument('--class_num', type=int, default=7)
  argparser.add_argument('--features_num', type=int, default=1433)
  argparser.add_argument('--train_batch_size', type=int, default=140)
  argparser.add_argument('--val_batch_size', type=int, default=300)
  argparser.add_argument('--test_batch_size', type=int, default=1000)
  argparser.add_argument('--hidden_dim', type=int, default=128)
  argparser.add_argument('--depth', type=int, default=2)
  argparser.add_argument('--nbrs_num', type=int, default=100)
  argparser.add_argument('--drop_rate', type=float, default=0.0)
  argparser.add_argument('--learning_rate', type=float, default=0.01)
  argparser.add_argument('--epoch', type=int, default=60)
  argparser.add_argument('--client_num', type=int, default=0,
                         help="The number of graphlearn client on each pytorch worker, "
                              "which is used as `num_workers` of pytorh dataloader.")
  argparser.add_argument('--ddp', action="store_true", help="Whether use pytorch ddp.")
  args = argparser.parse_args()

  init_env(args)

  run(args)
