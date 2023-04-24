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

try:
  import torch
except ImportError:
  pass

import graphlearn as gl
import graphlearn.python.nn.pytorch as thg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TemporalBatch(object):
  def __init__(self, src, pos_dst, neg_dst, t, msg,
               n_id, edge_index, nbr_t, nbr_msg):
    """
    Args
      src: Source nodes of temporal batch events
      pos_dst: Dst nodes of temporal batch events
      neg_dst: Negative nodes for src
      t: Timestamps of events
      msg: Messages on events
      n_id: Unique ids in src, pos_dst, neg_dst and their neighbors
      edge_index: interactions between n_id(s)
      nbr_t: Timestamps of the interactions
      nbr_msg: Messages on the interactions
    """
    self.src = src
    self.pos_dst = pos_dst
    self.neg_dst = neg_dst
    self.t = t
    self.msg = msg
    self.n_id = n_id
    self.edge_index = edge_index
    self.nbr_t = nbr_t
    self.nbr_msg = nbr_msg
    self.num_events = src.size(0)

class TemporalBatchLoader(torch.utils.data.DataLoader):
  def __init__(self, graph, source, num_nodes,
               batch_size, nbr_size, msg_dim):
    self.num_nodes = num_nodes
    self.nbr_size = nbr_size
    self.msg_dim = msg_dim

    self.topo = torch.empty((num_nodes, nbr_size),
                            dtype=torch.long, device=device)

    events = graph.E(source).batch(batch_size).alias("event")
    query = self._sample_nbr(events)
    dataset = thg.Dataset(query, induce_func=self._induce_func)
    super().__init__(dataset, 1, shuffle=False, collate_fn=self)

  def __call__(self, batchs):
    return batchs[0]

  def _sample_nbr(self, events):
    srcV = events.outV().alias('src')
    dstV = events.inV().alias('pos_dst')
    negV = srcV.outNeg("interaction").sample(1).by("random").alias("neg_dst")
    srcV_nbr = srcV.outE("interaction").sample(
        self.nbr_size).by("topk").alias("src_nbr")
    dstV_nbr = dstV.inE("interaction").sample(
        self.nbr_size).by("topk").alias("dst_nbr")
    negV_nbr = negV.inE("interaction").sample(
        self.nbr_size).by("topk").alias("neg_nbr")
    return events.values()

  def _induce_func(self, data):
    """ Batch the data generated from GSL query as TemporalBatch.
    """
    for _, v in data.items():
      for name, elem in v.__dict__.items():
        if elem is not None and not isinstance(elem, dict):
          v.__dict__[name] = torch.from_numpy(elem).to(device)

    src = data['event'].ids
    pos_dst = data['event'].dst_ids
    neg_dst = data['neg_dst'].ids

    nbrs = ['src_nbr', 'dst_nbr', 'neg_nbr']
    self.topo[src], self.topo[pos_dst], self.topo[neg_dst] = [
      data[k].dst_ids.view(-1, self.nbr_size) for k in nbrs
    ]

    assoc_idx = torch.empty(self.num_nodes, dtype=torch.long, device=device)
    assoc_nodes = torch.empty(self.num_nodes, dtype=torch.long, device=device)
    n_id = torch.cat([src, pos_dst, neg_dst]).unique()
    neighbors = self.topo[n_id]

    nodes = n_id.view(-1, 1).repeat(1, self.nbr_size)
    mask = neighbors >= 0
    neighbors, nodes = neighbors[mask], nodes[mask]

    assoc_nodes[n_id] = torch.arange(n_id.size(0), device=n_id.device)
    nbr_msg = torch.cat([data[k].float_attrs for k in nbrs]
                        ).view(-1, self.nbr_size, self.msg_dim).to(device)
    nbr_t = torch.cat([data[k].timestamps for k in nbrs]
                      ).view(-1, self.nbr_size, 1).to(device)

    unique_nodes = assoc_nodes[n_id]
    nbr_msg = nbr_msg[unique_nodes][mask].view(-1, self.msg_dim)
    nbr_t = nbr_t[unique_nodes][mask].view(-1)

    n_id = torch.cat([n_id, neighbors]).unique()
    assoc_idx[n_id] = torch.arange(n_id.size(0), device=n_id.device)
    neighbors, nodes = assoc_idx[neighbors], assoc_idx[nodes]
    edge_index = torch.stack([neighbors, nodes])
    return TemporalBatch(src, pos_dst, neg_dst, data['event'].timestamps,
                         data['event'].float_attrs.view(-1, self.msg_dim),
                         n_id, edge_index, nbr_t, nbr_msg)
