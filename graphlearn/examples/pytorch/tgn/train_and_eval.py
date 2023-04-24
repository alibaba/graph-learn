# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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

import argparse
import os, sys
import time
import graphlearn as gl

try:
  import torch
except ImportError:
  pass
from torch.nn import Linear

from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
  IdentityMessage,
  LastAggregator
)

from sklearn.metrics import average_precision_score, roc_auc_score
from temporal_batch_loader import TemporalBatchLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphAttentionEmbedding(torch.nn.Module):
  def __init__(self, in_channels, out_channels, msg_dim, time_enc):
    super().__init__()
    self.time_enc = time_enc
    edge_dim = msg_dim + time_enc.out_channels
    self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                dropout=0.1, edge_dim=edge_dim)

  def forward(self, x, last_update, edge_index, t, msg):
    rel_t = last_update[edge_index[0]] - t
    rel_t_enc = self.time_enc(rel_t.to(x.dtype))
    edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
    return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
      super().__init__()
      self.lin_src = Linear(in_channels, in_channels)
      self.lin_dst = Linear(in_channels, in_channels)
      self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
      h = self.lin_src(z_src) + self.lin_dst(z_dst)
      h = h.relu()
      return self.lin_final(h)

class TGN(object):
  def __init__(self, graph, args):
    self.graph = graph
    self.args = args

    self.memory = TGNMemory(
      args.num_nodes,
      args.msg_dim,
      args.memory_dim,
      args.time_dim,
      message_module=IdentityMessage(
        args.msg_dim, args.memory_dim, args.time_dim),
      aggregator_module=LastAggregator(),
      ).to(device)

    self.gnn = GraphAttentionEmbedding(
      in_channels=args.memory_dim,
      out_channels=args.embedding_dim,
      msg_dim=args.msg_dim,
      time_enc=self.memory.time_enc,
      ).to(device)

    self.link_pred = LinkPredictor(in_channels=args.embedding_dim).to(device)
    self.criterion = torch.nn.BCEWithLogitsLoss()

    # Helper vector to map global node indices to local ones.
    self.assoc = torch.empty(args.num_nodes, dtype=torch.long, device=device)

  def optimizer(self):
    return torch.optim.Adam(
      set(self.memory.parameters()) | set(self.gnn.parameters())
      | set(self.link_pred.parameters()), lr=self.args.lr)

  def train(self):
    self.memory.train()
    self.gnn.train()
    self.link_pred.train()

    self.memory.reset_state()  # Start with a fresh memory.
    batch_loader = TemporalBatchLoader(
      self.graph, "train", self.args.num_nodes, self.args.batch_size,
      self.args.nbr_size, self.args.msg_dim)

    total_loss = 0
    total_size = 0
    optimizer = self.optimizer()
    for batch in batch_loader:
      optimizer.zero_grad()

      n_id = batch.n_id
      self.assoc[n_id] = torch.arange(n_id.size(0), device=device)

      # Get updated memory of all nodes involved in the computation.
      z, last_update = self.memory(n_id)

      z = self.gnn(z, last_update, batch.edge_index,
                   batch.nbr_t, batch.nbr_msg)

      pos_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.pos_dst]])
      neg_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.neg_dst]])

      loss = self.criterion(pos_out, torch.ones_like(pos_out))
      loss += self.criterion(neg_out, torch.zeros_like(neg_out))

      # Update memory with ground-truth state.
      self.memory.update_state(batch.src, batch.pos_dst, batch.t, batch.msg)

      loss.backward()
      optimizer.step()
      self.memory.detach()
      size = batch.num_events
      total_loss += float(loss) * size
      total_size += size

    return total_loss / total_size


  @torch.no_grad()
  def test(self, ds):
    self.memory.eval()
    self.gnn.eval()
    self.link_pred.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    batch_loader = TemporalBatchLoader(
      self.graph, ds, self.args.num_nodes, self.args.batch_size,
      self.args.nbr_size, self.args.msg_dim)

    aps, aucs = [], []
    for batch in batch_loader:
      n_id = batch.n_id

      self.assoc[n_id] = torch.arange(n_id.size(0), device=device)

      z, last_update = self.memory(n_id)
      z = self.gnn(z, last_update, batch.edge_index,
                   batch.nbr_t, batch.nbr_msg)

      pos_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.pos_dst]])
      neg_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.neg_dst]])

      y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
      y_true = torch.cat(
        [torch.ones(pos_out.size(0)),
          torch.zeros(neg_out.size(0))], dim=0)

      aps.append(average_precision_score(y_true, y_pred))
      aucs.append(roc_auc_score(y_true, y_pred))

      self.memory.update_state(batch.src, batch.pos_dst, batch.t, batch.msg)

    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())


def run(args):
  msg_dim = args.msg_dim
  data_folder = args.data_folder
  srcf, dstf, edgef, trainf, valf, testf = [
    os.path.join(data_folder, f) for f in args.files.split(',')]

  gl.set_default_neighbor_id(-1)
  gl.set_padding_mode(0)

  edge_decoder = gl.Decoder(attr_types=["float"] * msg_dim,
                            timestamped=True)

  g = gl.Graph()
  g = g.node(srcf, 'src', decoder=gl.Decoder()) \
      .node(dstf, 'dst', decoder=gl.Decoder()) \
      .edge(edgef, edge_type=("src", "dst", "interaction"),
            decoder=edge_decoder, directed=False) \
      .edge(trainf, edge_type=("src", "dst", "train"),
            decoder=edge_decoder, directed=False) \
      .edge(valf, edge_type=("src", "dst", "val"),
            decoder=edge_decoder, directed=False) \
      .edge(testf, edge_type=("src", "dst", "test"),
            decoder=edge_decoder, directed=False) \
      .init()

  tgn = TGN(g, args)
  for epoch in range(1, args.epoch):
    start = time.time()
    loss = tgn.train()
    duration = time.time() - start
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Time: {duration:.4f}s')
    val_ap, val_auc = tgn.test("val")
    test_ap, test_auc = tgn.test("test")
    print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')
    print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')


if __name__ == "__main__":
  cur_path = sys.path[0]
  argparser = argparse.ArgumentParser("Train TGN.")
  argparser.add_argument('--data_folder', type=str,
                         default=os.path.join(cur_path, '../../data/jodie/'),
                         help="Source dara folder, list files are node_table, edge_table, "
                              "train_table, val_table and test_table")
  argparser.add_argument('--files', type=str,
                         default="src,dst,wikipedia,wikipedia_train,wikipedia_val,wikipedia_test",
                         help="files names, join with `,`, in order of node_table, edge_table, "
                              "train_table, val_table and test_table")
  argparser.add_argument('--num_nodes', type=int, default=9227)
  argparser.add_argument('--msg_dim', type=int, default=172)
  argparser.add_argument('--batch_size', type=int, default=200)
  argparser.add_argument('--nbr_size', type=int, default=10)
  argparser.add_argument('--dropout', type=float, default=0.1)
  argparser.add_argument('--embedding_dim', type=int, default=100)
  argparser.add_argument('--memory_dim', type=int, default=100)
  argparser.add_argument('--time_dim', type=int, default=100)
  argparser.add_argument('--epoch', type=int, default=51)
  argparser.add_argument('--lr', type=float, default=0.0001)

  args = argparser.parse_args()
  run(args)